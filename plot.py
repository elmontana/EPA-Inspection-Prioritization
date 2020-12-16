import click
import numpy as np
import os
import pandas as pd
import pickle
import src.utils.data_utils as data_utils
import src.utils.plot_utils as plot_utils
import src.utils.sql_utils as sql_utils

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from src.evaluate.crosstabs import compute_crosstabs_for_models
from src.evaluate.model_selection import find_best_models



def plot_results_over_time(
    test_results_tables_prefix,
    metrics=['precision_score_at_600', 'num_labeled_samples_at_600'],
    base_rates=['precision_score_at_1.0', None],
    figsize=(20, 10), save_dir='./plots/'):
    """
    Plot test results of provided metrics, over time.

    Arguments:
        - test_results_tables_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metrics: a list of metrics (str) to plot results for
        - base_rates: a list of base rates, one for each metric
        - figsize: the size of the plotted figure
        - save_dir: directory where plots should be saved
    """
    plot_utils.plot_results_over_time(
        test_results_tables_prefix,
        metrics=metrics, base_rates=base_rates, save_dir=save_dir)


def plot_best_results_over_time(
    test_results_tables_prefix,
    metric='precision_score_at_600',
    other_metrics=['num_labeled_samples_at_600'],
    n=5,
    base_rates=['precision_score_at_1.0', None],
    figsize=(20, 10), save_dir='./plots/'):
    """
    Plot test results of best models at the provided metric, over time.

    Arguments:
        - test_results_tables_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metric: the metric to use for selecting best models and plotting results
        - other_metrics: any additional metrics to plot results for
        - n: number of models to plot
        - base_rates: a list of base rates, one for each metric
        - figsize: the size of the plotted figure
        - save_dir: directory where plots should be saved
    """
    best_model_idx = find_best_models(test_results_tables_prefix, metric=metric, n=n)
    plot_utils.plot_results_over_time(
        test_results_tables_prefix,
        metrics=([metric] + other_metrics), base_rates=base_rates,
        model_idx=best_model_idx, save_dir=save_dir)


def plot_best_precision_recall_curves(
    results_table_name,
    metric='precision_score_at_600', n=5, 
    include_baselines=True, save_dir='./plots/pr_k/'):
    """
    Plot precision recall curves for the best models.

    Arguments:
        - results_table_name: name of results table
        - metric: the metric to use for selecting best models
        - n: number of models to plot curves for
        - include_baselines: whether or not to also plot for baselines 
        - save_dir: directory where plots should be saved
    """
    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_table_prefix = results_table_name.split('_test_results')[0]
    results_table_prefix = results_table_prefix.rsplit('_', 1)[0]
    best_model_idx = find_best_models(results_table_prefix, metric=metric, n=n)
    if include_baselines:
        baseline_model_idx = data_utils.get_baseline_model_idx(results_table_prefix)
        best_model_idx = list(best_model_idx) + list(baseline_model_idx)

    if not results_table_name.startswith('results.'):
        results_table_name = results_table_name.split('.')[-1]
        results_table_name = f'results.{results_table_name}'

    results_df = data_utils.get_table(results_table_name)
    results_df = results_df.iloc[best_model_idx]
    plot_utils.plot_pr_at_k(results_df, Path(save_dir) / 'curve')


def plot_best_feature_importances(
    exp_table_prefix, metric='precision_score_at_600', 
    n_models=5, n_features=12, include_baselines=True, save_dir='./plots/'):
    """
    Plot feature importances for the best models at the provided metric.

    Arguments:
        - exp_table_prefix: prefix of experiment tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metric: the metric to use for selecting best models
        - n_models: number of models to select
        - n_features: number of features to include in each plot
        - include_baselines: whether or not to also plot for baselines
        - save_dir: directory where plots should be saved
    """
    # Create save directories if they do not exist
    if not os.path.exists(Path(save_dir) / 'all_features'):
        os.makedirs(Path(save_dir) / 'all_features')
    if not os.path.exists(Path(save_dir) / f'{n_features}_features'):
        os.makedirs(Path(save_dir) / f'{n_features}_features')

    best_model_idx = find_best_models(exp_table_prefix, metric=metric, n=n_models)
    if include_baselines:
        baseline_model_idx = data_utils.get_baseline_model_idx(exp_table_prefix)
        best_model_idx = list(best_model_idx) + list(baseline_model_idx)

    # Get model paths
    test_results, _, _ = data_utils.get_test_results_over_time(exp_table_prefix)
    model_path_col_idx = test_results[0].columns.get_loc('model_path')
    model_paths = test_results[0].iloc[best_model_idx, model_path_col_idx].to_numpy()

    # Get feature names
    feature_names, feature_cols = data_utils.get_experiment_feature_names(exp_table_prefix)
    feature_names = np.array(feature_names)

    # Plot feature importances
    for path, model_idx in zip(model_paths, best_model_idx):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        
        feature_importance = np.array(model.feature_importance(columns=feature_cols))

        # Plot all features
        plot_utils.plot_feature_importances(
            feature_names, feature_importance, 
            Path(save_dir) / 'all_features'/ f'model_{model_idx}')

        # Plot most important features
        keep_idx = np.argsort(feature_importance)[::-1][:n_features]
        plot_utils.plot_feature_importances(
            feature_names[keep_idx], feature_importance[keep_idx], 
            Path(save_dir) / f'{n_features}_features'/ f'model_{model_idx}')


def plot_fairness_metric_over_groups(
    results_table_name, 
    fairness_metric='fdr', 
    feature_name='mean_county_income', feature_threshold=200000,
    performance_metric='precision_score_at_600', n_best_models=5,
    save_dir='./plots/',
    filename_prefix='model_disparity'):
    """
    Plot recall disparity scatter plot over groups.

    Arguments:
        - results_table_name: name of results table
        - fairness_metric: fairness metric, can be 'fdr' or 'tpr'
        - feature_name: feature name that is used to identify groups
        - feature_threshold: threshold to split the data into two groups
        - performance_metric: performance metric to plot on the x-axis
        - n_best_models: number of best models to highlight
        - save_dir: directory where plots should be saved
        - filename_prefix: prefix for the filename of the plot
    """
    plot_utils.plot_fairness_metric_over_groups(
        results_table_name,
        fairness_metric=fairness_metric, feature_name=feature_name, 
        pos_fn=lambda x: x >= feature_threshold, neg_fn=lambda x: x < feature_threshold,
        metric=performance_metric, n_best_models=n_best_models,
        save_dir=save_dir, filename_prefix=filename_prefix)


def compute_best_crosstabs(
    results_table_name, 
    metric='precision_score_at_600', n_models=5, 
    include_baselines=True, save_dir='./crosstabs/'):
    """
    Compute crosstabs for the best performing models.

    Arguments:
    
        - results_table_name: name of results table
        - metric: the metric to use for selecting best models
        - n_models: number of best models to select
        - include_baselines: whether or not to also plot for baselines
        - save_dir: directory where crosstabs should be saved
    """
    exp_table_prefix = results_table_name.split('_test_results')[0].rsplit('_', 1)[0]
    best_model_idx = find_best_models(exp_table_prefix, metric=metric, n=n_models)
    if include_baselines:
        baseline_model_idx = data_utils.get_baseline_model_idx(exp_table_prefix)
        best_model_idx = list(best_model_idx) + list(baseline_model_idx)

    k = int(metric.split('_')[-1])
    compute_crosstabs_for_models(best_model_idx, results_table_name, k=k, save_dir=save_dir)



@click.command()
@click.option('--exp_prefix', default='j_v1_model_grid_201212001909',
    help='prefix of experiment tables (e.g. "i_v1_test_run_201113235700")')
def main(exp_prefix):
    """
    Generate plots from an experiment.

    The plots are:
        - precision over time (all models)
        - precison over time (best models)
        - feature importances (best models)
        - fairness plots (fdr & tpr)

    Arguments:
        - exp_prefix: prefix of experiment tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
    """

    # Get name of lastest test result table
    split_date = lambda x: datetime.strptime(f'20{x[len(exp_prefix) + 1:].split("_")[0]}', '%Y%m%d')
    test_result_tables = data_utils.get_table_names(
        sql_utils.get_connection(), 'results', prefix=exp_prefix, suffix='test_results')
    test_results_table_name = sorted(test_result_tables, key=split_date)[-1]

    # Generate plots
    print('Plotting precision over time ...')
    plot_results_over_time(exp_prefix)

    print('Plotting precision for best 5 models over time ...')
    plot_best_results_over_time(exp_prefix, n=5)

    print('Plotting precision-recall curves for best 5 models from most recent year ...')
    plot_best_precision_recall_curves(test_results_table_name, n=5)

    print('Plotting feature importances for best 5 models ...')
    try:
        plot_best_feature_importances(exp_prefix, n_models=5, n_features=12)
    except PermissionError as e:
        print(e)

    print('Computing crosstabs for the best 5 models ...')
    compute_best_crosstabs(test_results_table_name, n_models=5)

    print('Plotting FDR & TPR fairness plots from most recent year ...')
    for metric in ['fdr', 'tpr']:
        plot_fairness_metric_over_groups(
            test_results_table_name,
            fairness_metric=metric,
            feature_name='mean_county_income',
            feature_threshold=100000,
            filename_prefix='rich_vs_poor')



if __name__ == '__main__':
    # So that every time we want to plot something,
    # we don't have to run main.py and spend an hour training models;
    # instead just use the results that are already in the database.
    main()
    
