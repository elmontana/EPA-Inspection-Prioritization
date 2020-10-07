import os
import pandas as pd
import pickle
import utils.sql_utils as sql

from sklearn.metrics import precision_score, recall_score, f1_score
from textwrap import dedent



def get_data(feature_table, label_table):
    """
    Get data from feature and label tables.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features

    Returns:
        - X: feature array
        - y: label array
    """
    
    # Query data from sql tables
    conn = sql.get_connection()
    sql_query = f'select f.*, l.label from {feature_table} f inner join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=conn)

    # Process data
    data = df.to_numpy(copy=True)
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y


def test(feature_table, label_table, model_paths=[], log_dir='./log_dir'):
    """
    Test models on validation data.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - model_paths: list of paths to the models being tested
        - log_dir: directory path for logging evaluation results
    """

    # Create log directory if not exists
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass

    # Process features and labels
    df = get_data(feature_table, label_table)
    

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

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        log_path = os.path.join(log_dir, f'{model_name}_eval.txt')
        with open(log_path, 'w') as log_file:
            log_file.writelines(dedent(log_text))



if __name__ == '__main__':
    feature_table = 'semantic.reporting'
    label_table = 'semantic.labels'
    model_dir = './saved_models'

    # Train models
    import model_train
    model_train.main(feature_table, label_table, model_train.load_grid_config(), model_dir)

    # Evaluate models
    model_paths = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.endswith('.pkl')]
    test(feature_table, label_table, model_paths)

