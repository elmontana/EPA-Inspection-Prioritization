import pandas as pd
from .sql_utils import get_connection



def get_data(feature_table, label_table, discard_columns=[]):
    """
    Get data from feature and label tables.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - discard_columns: list of column names to discard

    Returns:
        - X: feature data
        - y: label array
        - column_names: a list of feature column names
    """

    # Query data from sql tables
    conn = get_connection()
    sql_query = f'select f.*, l.label from {feature_table} f left join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=conn)
    df = df.drop(columns=discard_columns)

    # Process data
    data = df.to_numpy(copy=True)
    X, y = data[:, :-1], data[:, -1].astype(int)
    column_names = list(df.columns)[:-1]

    return X, y, column_names
