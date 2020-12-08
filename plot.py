import numpy as np
import pandas as pd
import pickle
import src.utils.data_utils as data_utils
import src.utils.plot_utils as plot_utils
import src.utils.sql_utils as sql_utils

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
    print(f'The best model indices are: {best_model_idx}.')

    plot_utils.plot_results_over_time(
        test_results_tables_prefix,
        metrics=([metric] + other_metrics), base_rates=base_rates,
        model_idx=best_model_idx, save_dir=save_dir)


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
        results_table_name = results_table_name.split('.')[-1]
        results_table_name = f'results.{results_table_name}'

    results_df = data_utils.get_table(results_table_name)
    results_df = results_df.iloc[best_model_idx]
    plot_utils.plot_pr_at_k(results_df, Path(save_dir) / 'curve')


def plot_best_feature_importances(
    exp_table_prefix, metric='precision_score_at_600', 
    n_models=5, n_features=12, save_dir='./plots/'):
    """
    Plot feature importances for the best models at the provided metric.

    Arguments:
        - exp_table_prefix: prefix of experiment tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metric: the metric to use for selecting best models
        - n_models: number of models to select
        - n_features: number of features to include in each plot
        - save_dir: directory where plots should be saved
    """
    best_model_idx = find_best_models(exp_table_prefix, metric=metric, n=n_models)

    # Get model paths
    test_results, _, _ = data_utils.get_test_results_over_time(exp_table_prefix)
    model_path_col_idx = test_results[0].columns.get_loc('model_path')
    model_paths = test_results[0].iloc[best_model_idx, model_path_col_idx].to_numpy()

    # Get feature names
    feature_names = data_utils.get_experiment_feature_names(exp_table_prefix)
    feature_names = np.array(feature_names)

    # Plot feature importances
    for path in model_paths:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        
        feature_importance = np.array(model.feature_importance())
        keep_idx = np.argsort(feature_importance)[::-1][:n_features]
        plot_utils.plot_feature_importances(feature_names[keep_idx], feature_importance[keep_idx], save_dir)


def plot_fairness_metric_over_groups(
    results_table_name, fairness_metric='fdr',
    feature_name='mean_county_income',
    pos_fn=lambda x: x > 200_000, neg_fn=lambda x: x <= 200_000,
    metric='precision_score_at_600', save_dir='./plots/',
    filename_prefix='model_disparity'):
    """
    Plot recall disparity scatter plot over groups.

    Arguments:
        - results_table_name: name of results table
        - fairness_metric: fairness metric, can be 'fdr' or 'tpr'
        - feature_name: feature name that is used to identify groups
        - feature_threshold: threshold to split the data to two groups
        - metric: the metric to use for metric axis
        - save_dir: directory where plots should be saved
        - filename_prefix: prefix for the filename of the plot
    """
    metric_k = metric.split('_at_')[-1]
    results_table_prefix = results_table_name.split('_test_results')[0]
    feature_table_name = f'experiments.{results_table_prefix}_test_features'
    label_table_name = f'experiments.{results_table_prefix}_test_labels'

    results_df = data_utils.get_table(f'results.{results_table_name}')
    feature_df = data_utils.get_table(feature_table_name)
    feature_df = feature_df[['entity_id', feature_name]]
    label_df = data_utils.get_table(label_table_name)

    num_models = len(results_df)
    model_classes = list(set(results_df['model_class'].to_list()))
    model_classes = list(sorted(model_classes, key=lambda s: s.split('.')[-1]))
    model_classes = model_classes[::-1]
    model_class_names = [s.split('.')[-1] for s in model_classes]
    model_metrics = results_df[metric].to_numpy()
    fairness_value = []
    for i in range(num_models):
        prediction_table_name = f'predictions.{results_table_prefix}_test_model_{i}'
        prediction_df = data_utils.get_table(prediction_table_name)
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
    fig = plt.figure(figsize=(8,6))
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
    for i in [102, 107, 113, 194, 196]:
        padding = 0.005 if fairness_metric == 'fdr' else 0.005
        plt.scatter([model_metrics[i]], [fairness_value[i]],
                    c='k', s=24)
        plt.text(s=f'Model {i}', ha='center', va='bottom',
                 x=model_metrics[i], y=fairness_value[i] + padding)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f'{filename_prefix}_{fairness_metric}.pdf')



if __name__ == '__main__':
    # So that every time we want to plot something,
    # we don't have to run main.py and spend an hour training models;
    # instead just use the results that are already in the database.

    test_results_tables_prefix = 'j_v1_model_grid_201203233617'

    #print('Plotting precision over time ...')
    #plot_results_over_time(test_results_tables_prefix)

    #print('Plotting precision for best 5 models over time ...')
    #plot_best_results_over_time(test_results_tables_prefix, n=5)

    print('Plotting feature importances for best 5 models ...')
    plot_best_feature_importances(test_results_tables_prefix, n_models=5, n_features=12)

    '''
    test_results_table_name = 'j_v1_model_grid_201203233617_160101_test_results'
    p10 = 49656.311
    p90 = 51535.599
    ref_group_fn = lambda x: np.logical_and(x > p10, x < p90)
    for metric in ['fdr', 'tpr']:
        plot_fairness_metric_over_groups(test_results_table_name,
                                         fairness_metric=metric,
                                         pos_fn=lambda x: x < p10,
                                         neg_fn=ref_group_fn,
                                         filename_prefix='p10_vs_middle')
        plot_fairness_metric_over_groups(test_results_table_name,
                                         fairness_metric=metric,
                                         pos_fn=lambda x: x > p90,
                                         neg_fn=ref_group_fn,
                                         filename_prefix='p90_vs_middle')
    '''

