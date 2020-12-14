import matplotlib
import numpy as np
import os
import pandas as pd
import random
import tqdm

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

from .sql_utils import get_connection
from .data_utils import get_test_results_over_time



### Helper Functions

def get_x_axis_values(columns, prefix, x_value_type):
    """
    Get x-axis values for precision-recall plots (i.e. k-values)
    from the column names of result table.

    Arguments:
        - columns: list of column names
        - prefix: prefix of columns to consider 
        - x_value_type: type for values; one of {'int', 'float'}

    Returns:
        - x_values: a list of x values
        - column_names: a list of column names
    """
    column_names = [col for col in list(columns) if col.startswith(prefix)]
    x_values_str = [s[len(prefix):] for s in column_names]

    if x_value_type == 'int':
        x_values_str = [s for s in x_values_str if not '.' in s]
        x_values = [int(s) for s in x_values_str]

    elif x_value_type == 'float':
        x_values_str = [s for s in x_values_str if '.' in s]
        x_values = [float(s) for s in x_values_str]

    else:
        raise ValueError('x_value type must be int or float.')

    return x_values, column_names



### Plotting

def plot_pr_at_k(
    results, save_prefix, 
    p_prefix='precision_score_at_', r_prefix='recall_score_at_', x_value_type='float'):
    """
    Plot precision recall curves.

    Arguments:
        - results: a results DataFrame, with a row for each model,
            and with columns for precision and recall at varying k-values
        - save_prefix: filename prefix to use when saving plots
        - p_prefix: the prefix for precision column names  
        - r_prefix: the prefix for recall column names  
        - x_value_type: type for k-values; one of {'int', 'float'}
    """
    # Get x-axis values from data frame
    p_x, p_cols = get_x_axis_values(results.columns, p_prefix, x_value_type)
    r_x, r_cols = get_x_axis_values(results.columns, r_prefix, x_value_type)

    for index, row in tqdm.tqdm(results.iterrows(), total=results.shape[0]):
        xlabel = 'k' if x_value_type == 'float' else 'n'
        p_values = [float(row[p_col]) for p_col in p_cols]
        r_values = [float(row[r_col]) for r_col in r_cols]

        fig, ax1 = plt.subplots()

        # Plot precision
        color = 'tab:red'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Precision', color=color)
        ax1.plot(p_x, p_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.0, 0.5)

        # Plot recall
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Recall', color=color)
        ax2.plot(r_x, r_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title(f'Precision Recall Curve for {row["model_class"]} {index}')
        
        # Save plot
        fig.tight_layout()
        plt.savefig(f'{save_prefix}_pr_at_k_model_{index}.png')
        plt.close(fig)


def plot_feature_importances(feature_names, feature_importance, save_dir):
    """
    Plot feature importances.
    
    Arguments:
        - feature_names: list of feature names
        - feature_importances: list of relative feature importance values
        - save_dir: directory where plot should be saved
    """
    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
                      
    assert len(feature_names) == len(feature_importance)
    y_pos = np.arange(len(feature_names))
    order = np.argsort(feature_importance)[::-1]
    feature_importance = feature_importance[order]
    feature_names = [feature_names[order[i]] for i in range(len(order))]

    fig, ax = plt.subplots()
    ax.barh(y_pos, feature_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    fig.set_size_inches(11.0, 8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'feature_importance.pdf'))
    plt.close(fig)


def plot_results_over_time(
    test_results_tables_prefix,
    metrics=['precision_score_at_600', 'num_labeled_samples_at_600'],
    base_rates=[0.02, None],
    model_idx=None,
    figsize=(20, 10), save_dir='./plots/'):
    """
    Plot test results of provided metrics, over time.

    Arguments:
        - test_results_tables_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metrics: a list of metrics (str) to plot results for
        - base_rates: a list of base rates, one for each metric
        - model_idx: a list of model indices to plot
        - figsize: the size of the plotted figure
        - save_dir: directory where plots should be saved
    """

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get test results, test dates, and model classes
    test_results, test_dates, model_classes = get_test_results_over_time(test_results_tables_prefix)

    # Filter model indices
    if model_idx is not None:
        test_results = [df.iloc[model_idx] for df in test_results]
        model_classes = [f'{model_classes[i]}_{i}' for i in model_idx]

    random.shuffle(model_classes)

    # Define a distinct color for each unique model class
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(model_classes))))
    colors = {model_class: color for model_class, color in zip(set(model_classes), colors)}
    colors['Base Rate'] = 'black'

    # Plot results over time for each metric
    for metric, base_rate in zip(metrics, base_rates):
        plt.clf()
        plt.figure(figsize=figsize)

        metric_idx = test_results[0].columns.get_loc(metric)
        metric_model_classes = model_classes.copy()

        # Plot all ML models
        for i, model_class in enumerate(metric_model_classes):
            if 'CommonSenseBaseline' not in model_class:
                results_over_time = [df.iloc[i, metric_idx] for df in test_results]
                plt.plot(test_dates, results_over_time, c=colors[model_class])

        # Plot common sense baselines
        for i, model_class in enumerate(metric_model_classes):
            if 'CommonSenseBaseline' in model_class:
                results_over_time = [df.iloc[i, metric_idx] for df in test_results]
                plt.plot(test_dates, results_over_time, c=colors[model_class])

        # Plot base rate
        if base_rate is not None:
            metric_model_classes.append('Base Rate')
            if isinstance(base_rate, str):
                base_rate_idx = test_results[0].columns.get_loc(base_rate)
                results_over_time = [df.iloc[0, base_rate_idx] for df in test_results]
                if metric == 'num_labeled_samples_at_600':
                    idx = test_results[0].columns.get_loc('num_labeled_rows')
                    scale = 600 / np.array([df.iloc[0, idx] for df in test_results])
                    results_over_time = scale * np.array(results_over_time)
                plt.plot(test_dates, results_over_time, c=colors['Base Rate'])
            else:
                plt.plot(test_dates, [base_rate] * len(test_dates), c=colors['Base Rate'])

        # Label axes and set title
        plt.xticks(test_dates)
        plt.xlabel('Evaluation Start Time')
        plt.ylabel(metric)
        plt.title(f'Model Group {metric} Over Time')

        # Create legend
        handles = [
            matplotlib.patches.Patch(color=colors[model_class], label=model_class)
            for model_class in set(metric_model_classes)]
        plt.legend(handles=handles)

        # Save plot
        plt.tight_layout()
        if model_idx is None:
            plt.savefig(Path(save_dir) / f'{metric}_plot.png')
        else:
            plt.savefig(Path(save_dir) / f'{metric}_plot_{len(model_idx)}_models.png')
