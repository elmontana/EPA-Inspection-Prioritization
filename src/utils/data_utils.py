import pandas as pd
from .sql_utils import get_connection


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
    conn = get_connection()
    sql_query = f'select f.*, l.label from {feature_table} f left join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=conn)

    # Process data
    data = df.to_numpy(copy=True)
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y
