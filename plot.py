import numpy as np
import src.utils.data_utils as data_utils
import src.utils.plot_utils as plot_utils

import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
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


def plot_best_precision_recall_curves(
    results_table_name,
    metric='precision_score_at_600', n=5, save_dir='./plots/'):
    """
    Plot precision recall curves for the best models.

    Arguments:
        - results_table_name: name of results table
        - metric: the metric to use for selecting best models
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


def plot_recall_disparity_over_groups(
    results_table_name,
    feature_name='county_population', feature_threshold=200_000,
    metric='precision_score_at_600', save_dir='./plots/'):
    """
    Plot recall disparity scatter plot over groups.

    Arguments:
        - results_table_name: name of results table
        - feature_name: feature name that is used to identify groups
        - feature_threshold: threshold to split the data to two groups
        - metric: the metric to use for metric axis
        - save_dir: directory where plots should be saved
    """
    metric_k = metric.split('_at_')[-1]
    results_table_prefix = results_table_name.split('_test_results')[0]
    feature_table_name = f'experiments.{results_table_prefix}_test_features'
    label_table_name = f'experiments.{results_table_prefix}_test_labels'

    results_df = data_utils.get_table(f'results.{results_table_name}')
    feature_df = data_utils.get_table(feature_table_name)
    group_ids = (feature_df[feature_name].to_numpy() > feature_threshold).astype(int)
    label_df = data_utils.get_table(label_table_name)

    num_models = len(results_df)
    model_classes = list(set(results_df['model_class']))
    model_metrics = results_df[metric].to_numpy()
    model_recall_disparities = []
    for i in range(num_models):
        prediction_table_name = f'predictions.{results_table_prefix}_test_model_{i}_pred_at_{metric_k}'
        prediction_df = data_utils.get_table(prediction_table_name)
        prediction_df['group_ids'] = group_ids
        combined_df = prediction_df.join(label_df.set_index('entity_id'), on='entity_id').dropna()
        predictions = combined_df['Prediction'].to_numpy()
        labels = combined_df['label'].to_numpy()
        gids = combined_df['group_ids'].to_numpy()

        # get group 0 recall
        tp0 = np.sum((predictions[gids == 0] == labels[gids == 0]) * (labels[gids == 0] == 1))
        pc0 = np.sum(labels[gids == 0])
        recall0 = tp0 / pc0

        # get group 1 recall
        tp1 = np.sum((predictions[gids == 1] == labels[gids == 1]) * (labels[gids == 1] == 1))
        pc1 = np.sum(labels[gids == 0])
        recall1 = tp1 / pc1

        model_recall_disparities.append(np.abs(recall0 - recall1))
    model_recall_disparities = np.array(model_recall_disparities)

    # prepare colors for the scatter plot
    vis_rgb = sns.color_palette("Set2", len(model_classes))
    colormap = dict(zip(list(range(len(model_classes))), vis_rgb))

    plt.clf()
    for i in range(len(model_classes)):
        model_indices = [k for k in range(len(model_classes)) \
                         if results_df.iloc[k]['model_class'] == model_classes[i]]
        plt.scatter(model_metrics[model_indices], model_recall_disparities[model_indices],
                    c=[colormap[i]], s=4)
    plt.xlabel(metric)
    plt.ylabel(f'Recall Disparity for Different [{feature_name}] Groups')
    plt.xlim(0, 0.25)
    plt.ylim(0, 1)
    plt.legend(model_classes)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'model_disparity.pdf')



if __name__ == '__main__':
    # So that every time we want to plot something,
    # we don't have to run main.py and spend an hour training models;
    # instead just use the results that are already in the database.

    '''
    print('Plotting precision over time ...')
    test_results_tables_prefix = 'i_v1_model_grid_201115015235'
    plot_results_over_time(test_results_tables_prefix)

    print('Plotting precision for best 5 models over time ...')
    test_results_tables_prefix = 'i_v1_model_grid_201115015235'
    plot_best_results_over_time(test_results_tables_prefix, n=5)

    print('Plotting precision recall curves for best 5 models over time ...')
    results_table_name = 'i_v1_model_grid_201115015235_160101_test_results'
    plot_precision_recall_curves(results_table_name)
    '''

    plot_recall_disparity_over_groups('i_v1_test_run_201113235700_160101_test_results')
