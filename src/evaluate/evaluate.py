import numpy as np
import os
import pandas as pd
import pickle

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..utils.data_utils import get_data



def get_predictions(model, X, k=None):
    """
    Get predictions from a model. 
    
    Arguments:
        - model: the trained model
        - X (np.ndarray): an array of features
        - k (int): the total number of positive labels we want to predict
    
    Returns:
        - y_pred: an array of label predictions
    """
    if k is None:
        return model.predict(X) > 0.5
    else:
        # Get the top-k highest predicted probabilities from the model
        probs = model.predict_proba(X)
        top_k = probs.argsort()[-k:][::-1]

        # Return an array of label predictions
        y_pred = np.zeros(len(probs))
        y_pred[top_k] = 1
        return y_pred


def evaluate(config, feature_table, label_table, model_paths, log_dir='./results/'):
    """
    Test models on validation data and save the results to a csv file.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
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
    metrics = [accuracy_score, precision_score, recall_score, f1_score]
    results = []

    for model_path in model_paths:
        # Load saved model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Evaluate predictions
        y_pred = get_predictions(model, X, k=700)
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

