import matplotlib
import numpy as np
import os
import pandas as pd
import random
import tqdm

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

from src.evaluate.model_selection import find_best_models
from .sql_utils import get_connection
from .data_utils import get_table, get_test_results_over_time
from ..evaluate.model_selection import find_best_models



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
    if x_value_type == 'int':
        column_names = [
            col for col in list(columns) 
            if col.startswith(prefix) and not '.' in col[len(prefix):]]
        x_values = [int(col[len(prefix):]) for col in column_names]

    elif x_value_type == 'float':
        column_names = [
            col for col in list(columns) 
            if col.startswith(prefix) and '.' in col[len(prefix):]]
        x_values = [float(col[len(prefix):]) for col in column_names]

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

    for index, row in results.iterrows():
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


def plot_feature_importances(feature_names, feature_importance, save_prefix):
    """
    Plot feature importances.
    
    Arguments:
        - feature_names: list of feature names
        - feature_importances: list of relative feature importance values
        - save_prefix: filename prefix to use when saving plots
    """
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
    fig.savefig(f'{save_prefix}_feature_importance.pdf')
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


def plot_fairness_metric_over_groups(
    results_table_name, fairness_metric='fdr',
    feature_name='mean_county_income',
    pos_fn=lambda x: x > 200_000, neg_fn=lambda x: x <= 200_000,
    metric='precision_score_at_600', n_best_models=5,
    save_dir='./plots/',
    filename_prefix='model_disparity'):
    """
    Plot recall disparity scatter plot over groups.

    Arguments:
        - results_table_name: name of results table
        - fairness_metric: fairness metric, can be 'fdr' or 'tpr'
        - feature_name: feature name that is used to identify groups
        - feature_threshold: threshold to split the data to two groups
        - metric: the metric to use for metric axis
        - n_best_models: number of best models to highlight
        - save_dir: directory where plots should be saved
        - filename_prefix: prefix for the filename of the plot
    """
    metric_k = metric.split('_at_')[-1]
    results_table_prefix = results_table_name.split('_test_results')[0]
    feature_table_name = f'experiments.{results_table_prefix}_test_features'
    label_table_name = f'experiments.{results_table_prefix}_test_labels'

    results_df = get_table(f'results.{results_table_name}')
    feature_df = get_table(feature_table_name, columns=['entity_id', feature_name])
    label_df = get_table(label_table_name)

    num_models = len(results_df)
    model_classes = list(set(results_df['model_class'].to_list()))
    model_classes = list(sorted(model_classes, key=lambda s: s.split('.')[-1]))
    model_classes = model_classes[::-1]
    model_class_names = [s.split('.')[-1] for s in model_classes]
    model_metrics = results_df[metric].to_numpy()
    fairness_value = []
    
    for i in tqdm.trange(num_models):
        prediction_table_name = f'predictions.{results_table_prefix}_test_model_{i}'
        prediction_df = get_table(
            prediction_table_name, columns=['entity_id', f'prediction_at_{metric_k}'])
        prediction_df = prediction_df.join(feature_df.set_index('entity_id'), on='entity_id').dropna()

        group_identifying_features = prediction_df[feature_name].to_numpy()
        group_ids = np.zeros_like(group_identifying_features) - 1
        group_ids[pos_fn(group_identifying_features)] = 1
        group_ids[neg_fn(group_identifying_features)] = 0
        prediction_df['group_ids'] = group_ids.astype(int)

        combined_df = prediction_df.join(label_df.set_index('entity_id'), on='entity_id').dropna()
        combined_df = combined_df[combined_df.group_ids != -1]
        predictions = combined_df[f'prediction_at_{metric_k}'].to_numpy()
        labels = combined_df['label'].to_numpy()
        gids = combined_df['group_ids'].to_numpy()

        if fairness_metric == 'tpr':
            tp0 = np.sum((predictions[gids == 0] == labels[gids == 0]) * (labels[gids == 0] == 1))
            pc0 = np.sum(labels[gids == 0])
            recall0 = tp0 / pc0

            tp1 = np.sum((predictions[gids == 1] == labels[gids == 1]) * (labels[gids == 1] == 1))
            pc1 = np.sum(labels[gids == 1])
            recall1 = tp1 / pc1

            fairness_value.append(recall1 / recall0)
        else:
            tp0 = np.sum((predictions[gids == 0] == labels[gids == 0]) * (labels[gids == 0] == 1))
            pc0 = np.sum(predictions[gids == 0] == 1)
            fdr0 = 1.0 - tp0 / pc0

            tp1 = np.sum((predictions[gids == 1] == labels[gids == 1]) * (labels[gids == 1] == 1))
            pc1 = np.sum(predictions[gids == 1] == 1)
            fdr1 = 1.0 - tp1 / pc1

            fairness_value.append(fdr1 / fdr0)
    fairness_value = np.array(fairness_value)

    # save recall disparity data to csv
    rd_df = pd.DataFrame({
        'index': list(range(num_models)),
        'model_class': [s.split('.')[-1] for s in results_df['model_class'].to_list()],
        metric: model_metrics,
        f'{fairness_metric}': fairness_value
    })
    rd_df.to_csv(Path(save_dir) / f'{filename_prefix}_{fairness_metric}.csv')

    # prepare colors for the scatter plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['.', '+', 'x', '^', 's']

    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(model_classes)):
        model_indices = [k for k in range(num_models) \
                         if results_df.iloc[k]['model_class'] == model_classes[i]]
        size = 24 if 'Common' in results_df.iloc[i]['model_class'] else 12
        plt.scatter(model_metrics[model_indices], fairness_value[model_indices],
                    c=[colors[i]], s=size, marker=markers[i])
    plt.xlabel(metric)
    plt.ylabel(f'{fairness_metric.upper()} Disparity for Different [{feature_name}] Groups')
    # plt.xlim(0, 0.1)
    # if fairness_metric == 'fdr':
    #     plt.ylim(0.8, 1.2)
    plt.legend(model_class_names)

    best_model_idx = find_best_models(results_table_prefix, metric=metric, n=n_best_models)
    for i in best_model_idx:
        padding = 0.005 if fairness_metric == 'fdr' else 0.005
        plt.scatter([model_metrics[i]], [fairness_value[i]],
                    c='k', s=24)
        plt.text(s=f'Model {i}', ha='center', va='bottom',
                 x=model_metrics[i], y=fairness_value[i] + padding)

    model_classes = results_df['model_class'].to_numpy()
    baseline_idx = [
        i for i in range(len(model_classes)) 
        if model_classes[i].rsplit('.', 1)[-1] == 'CommonSenseBaseline']
    for i in baseline_idx:
        padding = 0.005 if fairness_metric == 'fdr' else 0.005
        plt.scatter([model_metrics[i]], [fairness_value[i]],
                    c='k', s=24)
        plt.text(s=f'Baseline {i}', ha='center', va='bottom',
                 x=model_metrics[i], y=fairness_value[i] + padding)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / f'{filename_prefix}_{fairness_metric}.png')
