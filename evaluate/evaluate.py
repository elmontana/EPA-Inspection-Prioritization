import numpy as np
import os
import pandas as pd
import pickle

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_utils import get_data



def evaluate(config, feature_table, label_table, model_paths, log_dir='./results/'):
    """
    Test models on validation data.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - log_dir: directory for saving evaluation results
    
    Returns:
        - list of evaluation metrics for each model tested
    """

    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get feature and label arrays
    X, y = get_data(feature_table, label_table)

    # Evaluate models
    metrics = [accuracy_score, precision_score, recall_score, f1_score]
    results = []

    for model_path in model_paths:
        # Load saved model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Evaluate predictions
        y_pred = model.predict(X) > 0.5
        model_results = [metric(y, y_pred) for metric in metrics]
        results.append(model_results)

    # Convert results to dataframe table
    columns = [metric.__name__ for metric in metrics]
    results = pd.DataFrame(np.array(results), index=model_paths, columns=columns)

    # Save results to csv file
    experiment_name = config['experiment_name']
    results_path = Path(log_dir) / f'{experiment_name}_results.csv'
    results.to_csv(results_path)

    return results

