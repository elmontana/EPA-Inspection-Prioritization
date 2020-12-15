import click
import getpass
import glob
import os
import tqdm
import yaml
import pickle
import numpy as np
import pandas as pd

import src.utils.data_utils as data_utils
import src.utils.date_utils as date_utils
import src.utils.plot_utils as plot_utils
import src.utils.sql_utils as sql_utils

from src.preprocessing.run import main as run_preprocess
from src.model_prep.cohorts import prepare_cohort
from src.train import train
from src.evaluate import evaluate, get_predictions

import warnings
warnings.filterwarnings('ignore', category=UserWarning)



def parse_temporal_config(temporal_config):
    """
    Takes a config file and returns two lists of dictionaries for each iteration of
    model training/testing. One list is for the test set, the other is for the training set.
    Each dictionary contains critical feature and label start and endtimes.

    Arguments:
        temporal_config: config file with labels for
            feature_start_time, feature_duration, label_duration, and the train_repeat_interval

    Returns:
        train_splits: list of dictionaries of feature/label start/endtimes for various training sets
        test_splits: list of dictionaries of feature/label start/endtimes for various testing sets
    """

    # Convert dates and time invervals from temporal_config into `datetime` objects
    feature_start = date_utils.parse_date(temporal_config['feature_start_time'])
    feature_duration = date_utils.parse_interval(temporal_config['feature_duration'])
    label_duration = date_utils.parse_interval(temporal_config['label_duration'])
    repeat_interval = date_utils.parse_interval(temporal_config['train_repeat_interval'])
    feature_aod_interval = date_utils.parse_interval(temporal_config['feature_aod_interval'])

    train_splits = []
    test_splits = []

    # For every training instance, create a dictionary of start and end times
    # for the training and testing data
    for i in range(temporal_config['num_train_repeat']):
        train_start_time = feature_start + repeat_interval * i
        train_split = []

        curr_aod = train_start_time + feature_aod_interval
        while curr_aod <= train_start_time + feature_duration:
            train_split.append({
                'feature_start_time': feature_start,
                'feature_end_time': curr_aod,
                'label_start_time': curr_aod,
                'label_end_time': curr_aod + label_duration
            })
            curr_aod += feature_aod_interval
        train_splits.append(train_split)

        test_start_time = train_start_time + label_duration
        test_splits.append({
            'feature_start_time': test_start_time,
            'feature_end_time': test_start_time + feature_duration,
            'label_start_time': test_start_time + feature_duration,
            'label_end_time': test_start_time + feature_duration + label_duration
        })

    return train_splits, test_splits



def compute_crosstab_for_model(model, X, y, feature_names, save_prefix, metric, k):
    """
    Computes crosstab for a given model.

    Arguments:
        model: object for the model.
        X: features to compute crosstab on.
        y: labels to compute crosstab on.
        feature_names: a list of string corresponding to feature names.
        save_prefix: prefix for saving to database.
        metric: the metric to be used for prediction.
        k: the k value to be used for prediction.
    """
    assert type(k) == int, 'Crosstab only supports integer k value now.'
    assert len(feature_names) == X.shape[1], 'Feature name and X do not match.'
    y_pred, probs = get_predictions(model, X, k_values=[k])
    feature_means, feature_stds = X.mean(axis=0), X.std(axis=0)
    pos_pred_means = X[y_pred == 1].mean(axis=0)
    neg_pred_means = X[y_pred == 0].mean(axis=0)
    pos_pred_mean_z = (pos_pred_means - feature_means) / (feature_stds + 1e-12)
    neg_pred_mean_z = (neg_pred_means - feature_means) / (feature_stds + 1e-12)
    z_diff = np.abs(pos_pred_mean_z - neg_pred_mean_z)
    z_diff_desc = np.argsort(z_diff)[::-1]
    num_features = len(feature_names)
    crosstab_data = {
        'feature_name': [feature_names[z_diff_desc[i]] for i in range(num_features)],
        'positive_means': pos_pred_means[z_diff_desc],
        'negative_means': neg_pred_means[z_diff_desc],
        'normalized_difference': z_diff[z_diff_desc],
    }
    crosstab_df = pd.DataFrame(crosstab_data)
    crosstab_df.to_sql(save_prefix, sql_utils.get_connection(),
                       schema='predictions', index=True)



@click.command()
@click.option('--config', default='experiments/test_run.yaml',
    help='Path to config file.')
@click.option('--run_preprocessing', is_flag=True,
    help='Whether or not to run the preprocessing step.')
@click.option('--run_data_upload', is_flag=True,
    help='Whether or not to run data upload to DB before preprocessing.')
@click.option('--log_dir', type=str, default='logs',
    help='Directory to save trained model and testing results.')
def main(config, run_preprocessing, run_data_upload, log_dir):
    # Load experiment configuration
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get db connection
    conn = sql_utils.get_connection()

    # Get basic info of experiment
    exp_version = config['version']
    exp_name = config["experiment_name"]
    exp_time = date_utils.get_current_time_string()[2:]
    username = getpass.getuser()[0]

    terminal_width = int(os.popen('stty size', 'r').read().split()[1])
    print(f'Running Experiment: {username}_{exp_version}_{exp_name}_{exp_time}\n{"-" * terminal_width}\n')

    # Preprocessing
    preprocessing_prefix = config['preprocessing_config']['prefix']
    if not run_preprocessing:
        print('Preprocessing skipped.')
    else:
        print('Preprocessing ...')
        run_preprocess(conn, config['preprocessing_config'],
                       run_data_upload=run_data_upload)
        print('Preprocessing done.')

    # Get temporal configuration information
    train_dates_list, test_dates_list = parse_temporal_config(config['temporal_config'])

    # Training and evaluation
    test_results_over_time = []
    experiment_loop = tqdm.tqdm(list(zip(train_dates_list, test_dates_list)), desc='Experiment Repeats')
    for train_dates, test_dates in experiment_loop:
        split_time_abbr = date_utils.date_to_string(test_dates['label_start_time'])
        split_time_abbr = split_time_abbr.replace('-', '')[2:]
        split_name = f'{split_time_abbr}'
        print(split_name)
        prefix = f'{username}_{exp_version}_{exp_name}_{exp_time}_{split_name}'
        experiment_table_prefix = f'experiments.{prefix}'
        train_save_dir = os.path.join(
            os.getcwd(), log_dir, prefix, 'train_' + exp_time)
        test_save_dir = os.path.join(
            os.getcwd(), log_dir, prefix, 'test_' + exp_time)

        # Prepare cohort as specified by our experiment configuration
        tqdm.tqdm.write('\nPreparing cohorts ...')
        train_feature_splits, train_label_splits = [], []
        for i, train_dates_aod in enumerate(train_dates):
            train_feature_table, train_label_table = prepare_cohort(
                config,
                train_dates_aod, test_dates,
                preprocessing_prefix, experiment_table_prefix + f'_split{i}',
                include_test=False)[:2]
            train_feature_splits.append(train_feature_table)
            train_label_splits.append(train_label_table)
        test_feature_table, test_label_table = prepare_cohort(
            config,
            train_dates[-1], test_dates,
            preprocessing_prefix, experiment_table_prefix,
            include_train=False)[2:]
        train_feature_table = f'{experiment_table_prefix}_train_features'
        sql_utils.merge_tables(train_feature_splits, train_feature_table)
        train_label_table = f'{experiment_table_prefix}_train_labels'
        sql_utils.merge_tables(train_label_splits, train_label_table)

        # Delete intermediate cohort tables
        for i in range(len(train_dates)):
            cohort_table_name = f'{experiment_table_prefix}_split{i}_cohort'
            sql_utils.run_sql_from_string(conn, f'drop table {cohort_table_name};')

        # Train models as specified by our experiment configuration
        tqdm.tqdm.write('Training ...')
        model_summaries = train(
            config,
            train_feature_table, train_label_table,
            discard_columns=['split'],
            save_dir=train_save_dir)

        # Evaluate our models on the training data
        model_paths = glob.glob(f'{train_save_dir}/*.pkl')
        tqdm.tqdm.write('Evaluating on training data ...')
        train_results = evaluate(
            config,
            train_feature_table, train_label_table,
            model_paths, model_summaries,
            discard_columns=['split'],
            log_dir=train_save_dir)

        # Evaluate our models on the test data
        tqdm.tqdm.write('Evaluating on test data ...')
        test_results = evaluate(
            config,
            test_feature_table, test_label_table,
            model_paths, model_summaries,
            save_preds_to_db=True,
            save_prefix=f'{prefix}_test',
            log_dir=test_save_dir)
        test_results_over_time.append(test_results)

        # Use the first metric to find the best model
        num_metrics = len(config['eval_config']['metrics'])
        num_k = len(config['eval_config']['k'])
        result_columns = test_results.columns
        model_metrics = test_results[result_columns[-num_metrics * num_k]]
        sorted_model_index = np.argsort(model_metrics.to_list())
        for ranking, model_index in enumerate(sorted_model_index):
            if test_results['model_class'][model_index].endswith('CommonSenseBaseline'):
                print(f'The model ranked #{ranking} is a baseline, skipping.')
                continue
            else:
                plot_utils.plot_pr_at_k(test_results.iloc[model_index].to_frame().T, "best_p_at_600")
                best_model_path = test_results['model_path'][model_index]
                with open(best_model_path, 'rb') as f:
                    model = pickle.load(f)
                feature_importance = model.feature_importance()
                print(f'Plotting feature importance for model #{model_index}.')
                break
        feature_names = sql_utils.get_table_columns(conn, train_feature_table)[2:]
        plot_utils.plot_feature_importances(feature_names, feature_importance,
                                            test_save_dir)

        # Calculate crosstab for the model
        X, y = data_utils.get_data(test_feature_table, test_label_table)
        k = config['eval_config']['k'][0]
        compute_crosstab_for_model(model, X, y, feature_names,
                                   f'{prefix}_best_model_crosstab_at_{k}',
                                   config['eval_config']['metrics'][0], k)

        # Save results to database
        train_results_name = f'{prefix}_train_results'
        test_results_name = f'{prefix}_test_results'
        train_results.to_sql(train_results_name, conn, schema='results')
        test_results.to_sql(test_results_name, conn, schema='results')

    # Plot test results over time
    test_results_tables_prefix = f'{username}_{exp_version}_{exp_name}_{exp_time}'
    plot_utils.plot_results_over_time(test_results_tables_prefix)



if __name__ == '__main__':
    main()
