import importlib
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool


from pathlib import Path
from ..models.wrappers import SKLearnWrapper
from ..utils.data_utils import get_data
from ..utils.plot_utils import plot_metric_at_k, plot_pr_at_k
from ..utils.sql_utils import get_connection



def get_predictions(model, X, k=None, columns=None, save_db_table=None):
    """
    Get predictions from a model.

    Arguments:
        - model: the trained model
        - X (np.ndarray): an array of features
        - k (float or int): the total number of positive labels we want to predict
            If provided as a float within (0.0, 1.0), k is the total proportion of positive labels
        - columns (list): list of column names of the features
        - entity_ids: list of entity ids
        - save_db_table (str): name of table in which to save predictions

    Returns:
        - y_pred: an array of label predictions
        - probs: the probabilities for each prediction
    """

    # Wrap sklearn models
    if model.__module__.startswith('sklearn'):
        model = SKLearnWrapper(model)

    # Get probabilities
    probs = model.predict_proba(X.to_numpy(copy=True), columns=list(X.columns))[:, 1]

    if k is None:
        y_pred = probs > 0.5
        return y_pred, probs

    if isinstance(k, float):
        # Convert k from proportion to an integer number of positive labels
        k = int(float(len(probs)) * k)

    # Create an array of label predictions
    top_k_indices = probs.argsort()[-k:][::-1]
    y_pred = np.zeros(len(probs))
    y_pred[top_k_indices] = 1

    # Save predictions to database
    if save_db_table is not None:
        data = np.stack([y_pred, probs], axis=-1)
        data = pd.DataFrame(index=X.index, data=data, columns=['Prediction', 'Probability'])
        data.to_sql(save_db_table, get_connection(), schema='predictions', index=True)

    return y_pred, probs


def evaluate_single_model_single_arg(arg):
    """
    Evaluate a single model with provided model specifications and data, using a
    single argument to fit the imap interface.

    Arguments:
        - arg: a tuple that includes arguments to a evaluate_single_model call.
    """
    return evaluate_single_model(*arg)


def evaluate_single_model(save_prefix, save_preds_to_db, i, model_path,
                          k_values, X, y, labeled_indices, metrics):
    """
    Evaluate a single model with provided model specifications and data.

    Arguments:
        - save_prefix: prefix for saving the results
        - save_preds_to_db: whether to save predictions to database
        - i: index for the model
        - model_path: path to load the model
        - k_values: k values used for computing the metrics
        - X: features
        - y: labels
        - labeled_indices: indices that have labels
        - metrics: a list of metrics to use
    """
    # Load saved model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Evaluate predictions
    model_results = []
    for k in k_values:
        save_db_table = f'{save_prefix}_model_{i}_pred_at_{k}' if save_preds_to_db else None
        y_pred, probs = get_predictions(model, X, k=k, save_db_table=save_db_table)
        y_pred_filtered = y_pred[labeled_indices]
        y_filtered = y.to_numpy(copy=True)[labeled_indices]
        model_results.extend([metric(y_filtered, y_pred_filtered) for metric in metrics])

    return i, model_results


def evaluate(
    config, feature_table, label_table,
    model_paths, model_configs,
    save_preds_to_db=False, save_prefix='',
    discard_columns=[], log_dir='./results/'):
    """
    Test models on validation data and save the results to a csv file.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - model_configs: list of dictionaries containing model configs
        - save_preds_to_db: whether or not to save predictions to database
        - save_prefix: string prefix for any tables created
        - discard_columns: names of columns to discard before building matrices
        - log_dir: directory for saving evaluation results

    Returns:
        - results: a DataFrame containing the results of the
            evaluation metrics for each model
    """

    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get feature and label data
    X, y = get_data(feature_table, label_table, discard_columns=discard_columns)
    labeled_indices = np.logical_or(y == 0, y == 1)

    # Evaluate models
    metrics_str = [s.rsplit('.', 1) for s in config['eval_config']['metrics']]
    metrics = [getattr(importlib.import_module(m), c) for (m, c) in metrics_str]
    k_values = config['eval_config']['k']

    pool = Pool(processes=5)
    num_models = len(model_paths)
    args = zip(repeat(save_prefix, num_models),
               repeat(save_preds_to_db, num_models), list(range(num_models)),
               model_paths, repeat(k_values, num_models),
               repeat(X, num_models), repeat(y, num_models),
               repeat(labeled_indices, num_models),
               repeat(metrics, num_models))
    results = [None] * num_models
    for _ in tqdm(pool.imap(evaluate_single_model_single_arg, args),
                  total=num_models, desc='Evaluate models'):
        i, model_results = _
        results[i] = model_results
    pool.close()

    # Convert results to dataframe table
    columns = [[f'{metric.__name__}_at_{k}' for metric in metrics] for k in k_values]
    columns = [item for sublist in columns for item in sublist]
    model_configs = pd.DataFrame(model_configs)
    model_filenames = pd.DataFrame({ 'model_path': model_paths })
    info = pd.DataFrame({ 'num_labeled_rows': [int(labeled_indices.sum())] * len(model_paths) })
    results = pd.DataFrame(np.array(results).round(4), columns=columns)
    results = pd.concat([model_configs, model_filenames, info, results], axis=1)

    # Save results to csv file
    experiment_name = config['experiment_name']
    results_path = Path(log_dir) / f'{experiment_name}_results.csv'
    results.to_csv(results_path)

    # Check if model results include precision and recall
    metric_includes_precision = False
    metric_includes_recall = False
    for s in columns:
        if s.startswith('precision_score_at_'):
            metric_includes_precision = True
        if s.startswith('recall_score_at_'):
            metric_includes_recall = True

    # Plot precision@k curve
    if metric_includes_precision:
        save_path = Path(log_dir) / f'{experiment_name}_precision_at_k.pdf'
        plot_metric_at_k(results, prefix='precision_score_at_',
                         x_value_type='float',
                         save_path=save_path)

    # Plot recall@k curve
    if metric_includes_recall:
        save_path = Path(log_dir) / f'{experiment_name}_recall_at_k.pdf'
        plot_metric_at_k(results, prefix='recall_score_at_',
                         x_value_type='float',
                         save_path=save_path)

    # Plot pr@k for all models
    if metric_includes_precision and metric_includes_recall:
        plot_pr_at_k(results, 'float', 'precision_score_at_', 'recall_score_at_',
                     Path(log_dir) / experiment_name)


    return results
