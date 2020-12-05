import itertools
import numpy as np
from ..utils.sql_utils import get_connection
from ..utils.data_utils import get_test_results_over_time



def find_best_models(table_prefix, metric='precision_score_at_600', sort_by='min', n=5):
    """
    Find the best n models given a set of test results over time.

    Arguments:
        - table_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metric: the metric we want to sort b
        - sort_by: the criteria we want to apply to our results over time {'min', 'average', 'last'}
        - n: the number of models to return

    Returns:
        - best_model_idx: the row indices of n the best models
    """
    test_results, _, _ = get_test_results_over_time(table_prefix)
    metrics = list(test_results[0].columns)
    assert metric in metrics

    filter_metric = f'num_labeled_samples_at_{metric.rsplit("_", 1)[-1]}'
    assert filter_metric in metrics


    def get_maximin_values(my_metric):
        # Create results matrix for our metric of shape (model, time)
        results_matrix = np.zeros((test_results[0].shape[0], len(test_results)))
        for i, result in enumerate(test_results):
            results_matrix[:, i] = result[my_metric].to_numpy()

        # Calculate mininum-values for our metric over time
        if sort_by == 'min':
            values = results_matrix.min(axis=-1)
        elif sort_by == 'average':
            values = results_matrix.mean(axis=-1)
        elif sort_by == 'last':
            values = results_matrix[:, -1]

        return values


    values = get_maximin_values(metric)

    # Filter out values where num_labeled_samples is below some threshold
    num_labeled_samples_min_threshold = 75
    num_labeled_samples_values = get_maximin_values(filter_metric)
    filter_idx = num_labeled_samples_values < num_labeled_samples_min_threshold
    values[filter_idx] = -1

    # Find the indices of the best models
    best_model_idx = values.argsort()[::-1]
    best_model_idx = best_model_idx[:n]
    print(best_model_idx, values[best_model_idx])

    return best_model_idx


def find_best_models_multiple(table_prefix, metrics=['precision_score_at_600'], sort_bys=['min'], n=1):
    """
    Find the best n models given a set of test results over time.

    Arguments:
        - table_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metrics: list of metrics we want to sort by
        - sort_bys: list of sort_by criteria
        - n: the number of models to return per configuration

    Returns:
        - best_model_idx: the row indices of the best models
    """
    grid = itertools.product(metrics, sort_bys)
    best_model_idx = []
    for metric, sort_by in grid:
        best_model_idx += find_best_models(table_prefix, metric=metric, sort_by=sort_by, n=n) 

    return best_model_idx

