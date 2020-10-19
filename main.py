import click
import getpass
import glob
import os
import yaml

import src.utils.date_utils as date_utils
import src.utils.sql_utils as sql_utils

from src.preprocessing.run import main as run_preprocess
from src.model_prep.cohorts import prepare_cohort
from src.train import train
from src.evaluate import evaluate



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

    train_splits = []
    test_splits = []
    
    # For every training instance, create a dictionary of start and endtimes
    # for the training and testing data
    for i in range(temporal_config['num_train_repeat']):
        train_start_time = feature_start + repeat_interval * i 
        train_splits.append({
            'feature_start_time': train_start_time,
            'feature_end_time': train_start_time + feature_duration,
            'label_start_time': train_start_time + feature_duration,
            'label_end_time': train_start_time + feature_duration + label_duration
        })

        test_start_time = train_start_time + label_duration
        test_splits.append({
            'feature_start_time': test_start_time,
            'feature_end_time': test_start_time + feature_duration,
            'label_start_time': test_start_time + feature_duration,
            'label_end_time': test_start_time+ feature_duration + label_duration
        })

    return train_splits, test_splits



@click.command()
@click.option('--config', default='experiments/test_run.yaml',
    help='Path to config file.')
@click.option('--skip_preprocessing', is_flag=True,
    help='Whether to skip the preprocessing step.')
@click.option('--log_dir', type=str, default='logs',
    help='Directory to save trained model and testing results.')
def main(config, skip_preprocessing, log_dir):
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

    # Preprocessing
    preprocessing_prefix = config['preprocessing_config']['prefix']
    if skip_preprocessing:
        print('Preprocessing skipped.')
    else:
        print('Preprosessing ...')
        run_preprocess(conn, config['preprocessing_config'])
        print('Preprocessing done.')

    # Get temporal configuration information
    train_dates_list, test_dates_list = parse_temporal_config(config['temporal_config'])

    # Training and evaluation
    for train_dates, test_dates in zip(train_dates_list, test_dates_list):
        split_time_abbr = date_utils.date_to_string(train_dates['feature_start_time'])
        split_time_abbr = split_time_abbr.replace('-', '')[2:]
        split_name = f'train_{split_time_abbr}'
        prefix = f'{username}_{exp_version}_{exp_name}_{exp_time}_{split_name}'
        experiment_table_prefix = f'experiments.{prefix}'
        train_save_dir = os.path.join(
            os.getcwd(), log_dir, prefix, 'train_' + exp_time)
        test_save_dir = os.path.join(
            os.getcwd(), log_dir, prefix, 'test_' + exp_time)

        # Prepare cohort as specified by our experiment configuration
        (train_feature_table, train_label_table, 
            test_feature_table, test_label_table) = prepare_cohort(
            config, 
            train_dates, test_dates, 
            preprocessing_prefix, experiment_table_prefix)

        # Train models as specified by our experiment configuration
        train(
            config, 
            train_feature_table, train_label_table,
            save_dir=train_save_dir)

        # Evaluate our models on the training data
        model_paths = glob.glob(f'{train_save_dir}/*.pkl')
        train_results = evaluate(
            config, 
            train_feature_table, train_label_table,
            model_paths, log_dir=train_save_dir)

        # Evaluate our models on the test data
        test_results = evaluate(
            config, 
            test_feature_table, test_label_table,
            model_paths, log_dir=test_save_dir)



if __name__ == '__main__':
    main()
