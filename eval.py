import os
import pandas as pd
import pickle
import utils.sql_utils as sql

from sklearn.metrics import precision_score, recall_score, f1_score
from textwrap import dedent


def query_table(table):
    """
    Queries the given table and returns a pandas DataFrame.

    Arguments:
        - table: the table name (e.g. 'semantic.labels')
    """
    conn = sql.get_connection()
    query_result = conn.execute(f'select * from {feature_table}')
    df = pd.DataFrame(query_result.fetchall())
    df.columns = query_result.keys()
    return df


def test_models(feature_table, label_table, config=None, log_dir='./'):
    """
    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - config: path to a configuration file
        - log_dir: path to directory for logging
    """

    # Process features and labels
    features, labels = query_table(feature_table), query_table(label_table)
    X, y_actual = features, labels # TODO: actually process features & labels into numpy arrays

    # Load saved model
    model_path = 'experiments/model.pkl' # TODO: get the actual path
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Evaluate predictions
    y_pred = model.predict(features)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)

    # Log results
    log_text = f'''
        Model Path: {model_path} 
        Config Path: {config}
        Precision: {precision}
        Recall: {recall}
        F1-Score: {f1}
    '''

    log_path = os.path.join(log_dir, '420.txt') # TODO: get the actual log path
    with open(log_path, 'w') as log_file:
        log_file.writelines(dedent(log_text))

