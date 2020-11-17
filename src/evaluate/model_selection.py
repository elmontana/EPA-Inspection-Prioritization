import numpy as np
from ..utils.sql_utils import get_connection
from ..utils.data_utils import get_test_results_over_time



def find_best_models(table_prefix, metric='precision_score_at_600', n=5):
    """
    Find the best n models given a set of test results over time.

    Arguments:
        - table_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metric: the metric we want to sort by
        - n: the number of models to return

    Returns:
        - best_model_idx: the row indices of n the best models
    """
    test_results, _, _ = get_test_results_over_time(table_prefix)
    metrics = list(test_results[0].columns)
    assert metric in metrics

    # Create results matrix for our metric of shape (model, time)
    results_matrix = np.zeros((test_results[0].shape[0], len(test_results)))
    for i, result in enumerate(test_results):
        results_matrix[:, i] = result[metric].to_numpy()

    # Calculate mininum-values for our metric over time
    mean_values = results_matrix.min(axis=-1)
    
    # Find the indices of the best models
    best_model_idx = mean_values.argsort()[::-1]
    best_model_idx = best_model_idx[:n]

    return best_model_idx

