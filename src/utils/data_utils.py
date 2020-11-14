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
    sql_query = f'select f.*, l.label from {feature_table} f left join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=get_connection())

    # Process data
    df = df.set_index('entity_id')
    df = df.drop(columns=discard_columns)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y


def get_table(table_name, columns=None):
    """
    Get data from SQL table as a pd.DataFrame object.

    Arguments:
        - table_name: name of table
        - columns: list of columns to select

    Returns:
        - df: a data frame with the given table columns
    """
    column_string = '*' if columns is None else ', '.join(columns)
    query = f'select {column_string} from {table_name}'
    df = pd.read_sql(query, con=get_connection())
    return df
