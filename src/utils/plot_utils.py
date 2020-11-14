import matplotlib
import numpy as np
import os
import pandas as pd
import tqdm

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

from .sql_utils import get_connection
from .data_utils import get_table



### Helper Functions

def get_x_axis_values(columns, prefix, x_value_type):
    column_names = [s for s in list(columns) if s.startswith(prefix)]
    x_values_str = [s[len(prefix):] for s in column_names]
    if x_value_type == 'int':
        x_values_str = [s for s in x_values_str if not '.' in s]
        x_values = [int(s) for s in x_values_str]
    elif x_value_type == 'float':
        x_values_str = [s for s in x_values_str if '.' in s]
        x_values = [float(s) for s in x_values_str]
    else:
        raise ValueError('x_value type must be int or float.')
    return x_values_str, x_values


def get_test_results_over_time(table_prefix):
    """
    Get data from test results over time for a single experiment run.

    Arguments:
        - table_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")

    Returns:
        - test_results: a list of pd.DataFrames, i.e. test results over time 
        - test_dates: list of test dates corresponding to test results
        - model_classes: a list of model classes (should be same across all data frames)
    """

    # Get names of test result tables
    query = f"select table_name from information_schema.tables where table_schema = 'results'"
    results_tables = pd.read_sql(query, con=get_connection()).to_numpy(copy=True).flatten()
    test_result_tables = [
        table for table in results_tables 
        if table.startswith(table_prefix) and table.endswith('test_results')]

    # Get corresponding data frames
    test_results = [get_table(f'results.{table}') for table in test_result_tables]

    # Get test dates & sort results by date
    test_dates = [int(f'20{table.split("_")[-3][:2]}') for table in test_result_tables]
    test_dates, test_results = zip(*sorted(zip(test_dates, test_results)))

    # Get names of model classes from data frames
    model_classes = test_results[0]['model_class'].to_numpy(copy=True)
    model_classes = [model_class.rsplit('.', 1)[-1] for model_class in model_classes]

    return test_results, test_dates, model_classes



### Plotting

def plot_metric_at_k(results, prefix, x_value_type='float', save_path=None):
    # clear figure
    plt.clf()

    # get x axis values from dataframe
    x_values_str, x_values = get_x_axis_values(results.columns, prefix,
                                               x_value_type)

    # iterate models and plot graphs
    for index, row in results.iterrows():
        y_values = [float(row[prefix + s]) for s in x_values_str]
        plt.plot(x_values, y_values)

    # add axis labels and save figure
    xlabel = 'k' if x_value_type == 'float' else 'n'
    ylabel = f'{prefix}{xlabel}'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([f'Model {i}' for i in range(len(results))])
    plt.tight_layout()
    plt.savefig(save_path)


def plot_pr_at_k(
    results, save_prefix, 
    p_prefix='precision_score_at_', r_prefix='recall_score_at_', x_value_type='float'):
    # get x axis values from dataframe
    p_xs, p_x = get_x_axis_values(results.columns, p_prefix, x_value_type)
    r_xs, r_x = get_x_axis_values(results.columns, r_prefix, x_value_type)

    for index, row in tqdm.tqdm(results.iterrows(), total=results.shape[0]):
        xlabel = 'k' if x_value_type == 'float' else 'n'
        p_values = [float(row[p_prefix + s]) for s in p_xs]
        r_values = [float(row[r_prefix + s]) for s in r_xs]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Precision', color=color)
        ax1.plot(p_x, p_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Recall', color=color)
        ax2.plot(r_x, r_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title(f'Precision Recall Curve for {row["model_class"]} {index}')
        
        fig.tight_layout()
        plt.savefig(f'{save_prefix}_pr_at_k_model_{index}.png')
        plt.close(fig)


def plot_feature_importances(feature_names, feature_importance, save_dir):
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
    metrics=['precision_score_at_600'], base_rates=[0.02],
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

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get test results, test dates, and model classes
    test_results, test_dates, model_classes = get_test_results_over_time(test_results_tables_prefix)

    # Define a distinct color for each unique model class
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(model_classes))))
    colors = {model_class: color for model_class, color in zip(set(model_classes), colors)}

    # Plot results over time for each metric
    plt.clf()
    plt.figure(figsize=figsize)
    for metric, base_rate in zip(metrics, base_rates):
        for i, model_class in enumerate(model_classes):
            results_over_time = [df.loc[i, metric] for df in test_results]
            plt.plot(test_dates, results_over_time, c=colors[model_class])

        # Plot base rate
        if base_rate is not None:
            colors['Base Rate'] = 'black'
            model_classes.append('Base Rate')
            plt.plot(test_dates, [base_rate] * len(test_dates), c=colors['Base Rate'])

        # Label axes and set title
        plt.xticks(test_dates)
        plt.xlabel('Evaluation Start Time')
        plt.ylabel(metric)
        plt.title(f'Model Group {metric} Over Time')

        # Create legend
        handles = [
            matplotlib.patches.Patch(color=colors[model_class], label=model_class)
            for model_class in set(model_classes)]
        plt.legend(handles=handles)

        # Save plot
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f'{metric}_plot.png')

    # Plot number of labeled samples over time
    num_labeled_samples = [results['num_labeled_samples'][0] for results in test_results]
    plt.clf()
    plt.plot(test_dates, num_labeled_samples)
    plt.xticks(test_dates)
    plt.xlabel('Evaluation Start Time')
    plt.ylabel('# of Labeled Samples')
    plt.title(f'Number of Labeled Samples Over Time')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'num_labeled_samples_plot.png')
