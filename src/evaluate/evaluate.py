import numpy as np
import os
import pandas as pd
import pickle
import importlib

from pathlib import Path
from ..utils.data_utils import get_data



def get_predictions(model, X, k=None, n=None):
    """
    Get predictions from a model. 
    
    Arguments:
        - model: the trained model
        - X (np.ndarray): an array of features
        - n (int): the total number of positive labels we want to predict
        - k (float): the total proportion of positive labels we want to predict
    
    Returns:
        - y_pred: an array of label predictions
    """
    assert not (k is not None and n is not None), 'k and n cannot be both specified.'
    if k is None and n is None:
        return model.predict(X) > 0.5
    else:
        # Get the top-k highest predicted probabilities from the model
        probs = model.predict_proba(X)
        if k is not None:
            n = int(float(len(probs)) * k)
        top_n = probs.argsort()[-n:][::-1]

        # Return an array of label predictions
        y_pred = np.zeros(len(probs))
        y_pred[top_n] = 1
        return y_pred, probs


def evaluate(config, feature_table, label_table, model_paths, model_configs,
             log_dir='./results/'):
    """
    Test models on validation data and save the results to a csv file.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - model_configs: list of dictionaries containing model configs
        - log_dir: directory for saving evaluation results
    
    Returns:
        - results: a DataFrame containing the results of the 
            evaluation metrics for each model
    """

    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get feature and label arrays
    X, y = get_data(feature_table, label_table)

    # Evaluate models
    metrics_str = [s.rsplit('.', 1) for s in config['eval_config']['metrics']]
    metrics = [getattr(importlib.import_module(m), c) for (m, c) in metrics_str]
    k_values = config['eval_config']['k']
    results = []

    for model_path in model_paths:
        # Load saved model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Evaluate predictions
        model_results = []
        for k in k_values:
            if type(k) is float:
                y_pred, probs = get_predictions(model, X, k=k)
            else:
                y_pred, probs = get_predictions(model, X, n=k)
            model_results.extend([metric(y, y_pred) for metric in metrics])
        results.append(model_results)

    # Convert results to dataframe table
    columns = [[f'{metric.__name__}_at_{k}' for metric in metrics] for k in k_values]
    columns = [item for sublist in columns for item in sublist]
    model_configs = pd.DataFrame(model_configs)
    model_filenames = pd.DataFrame({ 'model_path': model_paths })
    results = pd.DataFrame(np.array(results), columns=columns)
    results = pd.concat([model_configs, model_filenames, results], axis=1)

    # Save results to csv file
    experiment_name = config['experiment_name']
    results_path = Path(log_dir) / f'{experiment_name}_results.csv'
    results.to_csv(results_path)

    return results
