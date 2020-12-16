import numpy as np
import os
import pandas as pd

from pathlib import Path
from ..utils.data_utils import get_table
from ..utils.sql_utils import get_connection



def compute_crosstab_for_model(i, X, y_pred, k=600, save_dir='./crosstabs/'):
    """
    Computes crosstab for a given model.

    Arguments:
        - i: index of model 
        - X: data frame of features to compute crosstab on
        - y_pred: data frame of model predictions
        - k: the k-value to use for predictions
        - save_dir: directory where crosstabs should be saved
    """
    assert type(k) == int, 'Crosstab only support integer k values.'

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_pred = y_pred[f'prediction_at_{k}'].to_numpy().flatten()
    feature_means, feature_stds = X.mean(axis=0), X.std(axis=0)
    pos_pred_means = X[y_pred == 1].mean(axis=0)
    neg_pred_means = X[y_pred == 0].mean(axis=0)
    pos_pred_mean_z = (pos_pred_means - feature_means) / (feature_stds + 1e-12)
    neg_pred_mean_z = (neg_pred_means - feature_means) / (feature_stds + 1e-12)
    z_diff = np.abs(pos_pred_mean_z - neg_pred_mean_z)
    z_diff_desc = np.argsort(z_diff)[::-1]

    feature_names = np.array(X.columns)
    crosstab_data = {
        'feature_name': feature_names[z_diff_desc],
        'positive_means': pos_pred_means[z_diff_desc],
        'negative_means': neg_pred_means[z_diff_desc],
        'normalized_difference': z_diff[z_diff_desc],
    }

    # Save crosstabs as csv
    crosstab_df = pd.DataFrame(crosstab_data)
    crosstab_df.to_csv(Path(save_dir) / f'crosstab_model_{i}.csv')


def compute_crosstabs_for_models(model_idx, results_table_name, k=600, save_dir='./crosstabs/'):
    """
    Computes crosstab for the given models.

    Arguments:
        - model_idx: list of model indices
        - results_table_name: name of results table
        - k: the k-value to use for predictions
        - save_dir: directory where crosstabs should be saved
    """
    pred_table_prefix = results_table_name.split('_test_results')[0]
    X = get_table(f'experiments.{pred_table_prefix}_test_features')

    for i in model_idx:
        pred_table_name = f'predictions.{pred_table_prefix}_test_model_{i}'
        y_pred = get_table(pred_table_name, columns=[f'prediction_at_{k}'])
        compute_crosstab_for_model(i, X, y_pred, k=k, save_dir=save_dir)

