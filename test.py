import os
import pandas as pd
import pickle

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_utils import get_data



def test(
    feature_table, label_table, model_paths, 
    config_path='./experiments/test_run.yaml', log_dir='./log_dir'):
    """
    Test models on validation data.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - config_path: path to configuration file for this experiment
        - log_dir: directory for logging evaluation results
    """

    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

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
    results = pd.DataFrame(np.array(results), index=model_paths, columns=metrics)
    
    # Log results to csv file
    experiment_name = config['experiment_name']
    results_path = Path(log_dir) / f'{experiment_name}_results.csv'
    results.to_csv(results_path)    



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

