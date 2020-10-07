import os
import pandas as pd
import pickle

from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from textwrap import dedent
from utils.data_utils import get_data



def test(feature_table, label_table, model_paths, log_dir='./log_dir'):
    """
    Test models on validation data.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - log_dir: directory for logging evaluation results
    """

    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get feature and label arrays
    X, y = get_data(feature_table, label_table)
    
    # Evaluate models
    for model_path in model_paths:
        # Load saved model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Evaluate predictions
        y_pred = model.predict(X) > 0.5
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Log results
        log_text = f"""
            Model Path: {model_path} 
            Precision: {precision}
            Recall: {recall}
            F1-Score: {f1}
        """

        model_name = Path(model_path).stem
        log_path = Path(log_dir) / f'{model_name}_eval.txt'
        with open(log_path, 'w') as log_file:
            log_file.writelines(dedent(log_text))



if __name__ == '__main__':
    feature_table = 'semantic.reporting'
    label_table = 'semantic.labels'
    model_dir = './saved_models'

    # Train models
    import train
    train.train(feature_table, label_table, save_dir=model_dir)

    # Evaluate models
    model_paths = [Path(model_dir) / file for file in os.listdir(model_dir) if file.endswith('.pkl')]
    test(feature_table, label_table, model_paths)

