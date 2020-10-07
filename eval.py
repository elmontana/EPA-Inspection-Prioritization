import os
import pandas as pd
import pickle
import utils.sql_utils as sql

from sklearn.metrics import precision_score, recall_score, f1_score
from textwrap import dedent



def get_data(feature_table, label_table):
    """
    Queries the given tables and returns a pandas DataFrame.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
    """
    conn = sql.get_connection()
    sql_query = f'select f.*, l.label from {feature_table} f inner join {label_table} l on f.entity_id = l.entity_id;'
    return pd.read_sql(sql_query, con=conn)


def test_models(feature_table, label_table, model_paths=[], log_dir='./log_dir'):
    """
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
    data = df.to_numpy(copy=True)
    X, y = data[:, :-1], data[:, -1].astype(int)

    # Evaluate models
    for model_path in model_paths:

        # Load saved model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Evaluate models predictions
        y_pred = model.predict(X) > 0.5
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Log results
        log_text = f'''
            Model Path: {model_path} 
            Precision: {precision}
            Recall: {recall}
            F1-Score: {f1}
        '''

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        log_path = os.path.join(log_dir, f'{model_name}_eval.txt')
        with open(log_path, 'w') as log_file:
            log_file.writelines(dedent(log_text))



if __name__ == '__main__':
    feature_table = 'semantic.reporting'
    label_table = 'semantic.labels'

    # Train models
    import model_train
    model_train.main(feature_table, label_table, model_train.load_grid_config(), './saved_models')

    # Evaluate models
    model_paths = [os.path.join('./saved_models', file) for file in os.listdir('./saved_models') if file.endswith('.pkl')]
    test_models(feature_table, label_table, model_paths)

