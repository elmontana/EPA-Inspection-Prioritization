import numpy as np
import src.utils.data_utils as data_utils
import src.utils.plot_utils as plot_utils

from pathlib import Path
from src.evaluate.model_selection import find_best_models



def plot_results_over_time(
    test_results_tables_prefix, 
    metrics=['precision_score_at_600', 'num_labeled_samples_at_600'],
    base_rates=[0.02, None],
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
    base_rates=[0.02, None],
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


def plot_precision_recall_curves(results_table_name, save_dir='./plots/'):
    """
    Plot precision recall curves for each model in a set of results.

    Arguments: 
        - results_table_name: name of results table
        - save_dir: directory where plots should be saved
    """
    if not results_table_name.startswith('results.'):
        results_table_name = f'results.{results_table_name}'

    results_df = data_utils.get_table(results_table_name)
    plot_utils.plot_pr_at_k(results_df, Path(save_dir) / 'curve')


def plot_best_precision_recall_curves(results_table_name, n=5, save_dir='./plots/'):
    """
    Plot precision recall curves for the best models.

    Arguments: 
        - results_table_name: name of results table
        - n: number of models to plot curves for
        - save_dir: directory where plots should be saved
    """
    results_table_prefix = results_table_name.split('_test_results')[0]
    results_table_prefix = results_table_prefix.rsplit('_', 1)[0]
    best_model_idx = find_best_models(results_table_prefix, metric=metric, n=n)

    if not results_table_name.startswith('results.'):
        results_table_name = f'results.{results_table_name}'
    
    results_df = data_utils.get_table(results_table_name)
    results_df = results_df.iloc[best_model_idx]
    plot_utils.plot_pr_at_k(results_df, Path(save_dir) / 'curve')



if __name__ == '__main__':
    # So that every time we want to plot something, 
    # we don't have to run main.py and spend an hour training models;
    # instead just use the results that are already in the database.
    
    print('Plotting precision over time ...')
    test_results_tables_prefix = 'i_v1_test_run_201114232725'
    plot_results_over_time(test_results_tables_prefix)

    print('Plotting precision for best 5 models over time ...')
    test_results_tables_prefix = 'i_v1_test_run_201114232725'
    plot_best_results_over_time(test_results_tables_prefix, n=5)

    print('Plotting precision recall curves for best 5 models over time ...')
    results_table_name = 'i_v1_test_run_201114232348_120101_test_results'
    plot_best_precision_recall_curves(results_table_name, n=5)
