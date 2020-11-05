import pandas as pd
from .sql_utils import get_connection



def get_data(feature_table, label_table, discard_columns=[]):
    """
    Get data from feature and label tables as pd.DataFrame objects.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - discard_columns: list of column names to discard

    Returns:
        - X: feature data frame
        - y: label data frame
    """

    # Query data from sql tables
    conn = get_connection()
    sql_query = f'select f.*, l.label from {feature_table} f left join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=conn)

    # Process data
    df = df.set_index('entity_id')
    df = df.drop(columns=discard_columns)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y
