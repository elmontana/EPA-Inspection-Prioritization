import importlib
import numpy as np
import os
import pandas as pd
import pickle
import tqdm

from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

from ..utils.data_utils import get_data
from ..utils.plot_utils import plot_metric_at_k, plot_pr_at_k
from ..utils.sql_utils import get_connection



def get_predictions(model, X, k_values=[], columns=None, pred_table_name=None):
    """
    Get predictions from a model.

    Arguments:
        - model: the trained model
        - X: an array of features
        - k_values: a list of different k values;
            each k value is the total number of positive labels we want to predict;
            if provided as a float within (0.0, 1.0), k is the total proportion of positive labels
        - columns: list of column names of the features
        - pred_table_name: name of table in which to save predictions

    Returns:
        - y_preds: a (N x K) array of label predictions at each k value
        - probs: the prediction probabilities
    """

    # Get probabilities
    probs = model.predict_proba(X.to_numpy(copy=True), columns=list(X.columns))[:, 1]

    # Calculate predictions at each k value
    y_preds = np.zeros((len(probs), len(k_values)))
    for i, k in enumerate(k_values):
        if isinstance(k, float):
            # Convert k from proportion to an integer number of positive labels
            k = int(float(len(probs)) * k)

        # Make positive predictions for the top k probabilities
        top_k_indices = probs.argsort()[-k:][::-1]
        y_preds[top_k_indices, i] = 1

    # Save predictions to database
    if pred_table_name is not None:
        data = np.column_stack([y_preds, probs])
        data = pd.DataFrame(
            index=X.index, data=data, 
            columns=[*['Prediction at k={k}' for k in k_values], 'Probability'])
        data.to_sql(pred_table_name, get_connection(), schema='predictions', index=True)

    return y_preds, probs


def evaluate_single_model(
    model_path, model_index, save_preds_to_db, save_prefix,
    metrics, k_values, X, y, labeled_indices):
    """
    Evaluate a single model with provided model specifications and data.

    Arguments:
        - model_path: path to load the model
        - model_index: index for the model
        - save_preds_to_db: whether or not to save predictions to database
        - save_prefix: string prefix for any tables created
        - metrics: a list of metrics to use
        - k_values: k-values used for computing the metrics
        - X: feature array
        - y: label array
        - labeled_indices: indices of rows that have labels

    Returns:
        - model_index: index for the model
        - model_results: an (M x K) array of model results, for each metric, at each k-value
    """

    # Load saved model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Get predictions
    pred_table_name = f'{save_prefix}_model_{model_index}' if save_preds_to_db else None
    y_preds, probs = get_predictions(model, X, k_values=k_values, pred_table_name=pred_table_name)

    # Filter labels
    y_preds_filtered = y_preds[labeled_indices]
    y_filtered = y.to_numpy(copy=True)[labeled_indices]

    # Calculate metrics for each k value
    model_results = np.zeros((len(metrics), len(k_values)))
    for i, metric in enumerate(metrics):
        for j in range(len(k_values)):
            model_results[i, j] = metric(y_filtered, y_preds_filtered[:, j])

    return model_index, model_results


def evaluate_single_model_unpack_args(args):
    """
    Evaluate a single model with provided model specifications and data,
    using a single argument to fit the imap interface.

    Arguments:
        - args: a tuple with the arguments to an `evaluate_single_model` call.
    """
    return evaluate_single_model(*args)


def evaluate_multiprocessing(
    model_paths, save_preds_to_db, save_prefix,
    X, y, labeled_indices,
    metrics, k_values,
    num_processes=8):
    """
    Evaluate models in parallel.

    Arguments;
        - model_paths: list of paths to the models being tested
        - save_preds_to_db: whether or not to save predictions to database
        - save_prefix: string prefix for any tables created
        - X: feature array
        - y: label array
        - metrics: a list of metrics to use
        - k_values: k values used for computing the metrics
        - num_processes: number of different processes used for evaluation
    """
    num_models = len(model_paths)
    pool = Pool(processes=num_processes)
    args = zip(
        model_paths,
        range(num_models),
        repeat(save_preds_to_db, num_models),
        repeat(save_prefix, num_models),
        repeat(metrics, num_models),
        repeat(k_values, num_models),
        repeat(X, num_models),
        repeat(y, num_models),
        repeat(labeled_indices, num_models))

    results = [None] * num_models
    for model_index, model_results in tqdm.tqdm(
        pool.imap(evaluate_single_model_unpack_args, args),
        total=num_models, desc='Evaluating models'):

        results[model_index] = model_results.flatten()

    pool.close()
    return results


def evaluate(
    config, feature_table, label_table,
    model_paths, model_summaries,
    save_preds_to_db=False, save_prefix='',
    discard_columns=[], log_dir='./results/'):
    """
    Test models on validation data and save the results to a csv file.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - model_summaries: list of model summary dictionaries
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
    results = evaluate_multiprocessing(
        model_paths, save_preds_to_db, save_prefix,
        X, y, labeled_indices, metrics, k_values)

    # Convert results to dataframe table
    results_columns = [f'{metric.__name__}_at_{k}' for metric in metrics for k in k_values]
    results = pd.DataFrame({
        **pd.DataFrame(model_summaries),
        'model_path': model_paths,
        'num_labeled_rows': [int(labeled_indices.sum())] * len(model_paths),
        **pd.DataFrame(np.array(results).round(4), columns=results_columns),
    })

    # Save results to csv file
    experiment_name = config['experiment_name']
    results_path = Path(log_dir) / f'{experiment_name}_results.csv'
    results.to_csv(results_path)

    return results
