import os
import pandas as pd
import sqlalchemy



### Constants

DB = 'epa3_database'



### SQL Utility Functions

def get_credentials(path='~/.pgpass', db=DB):
    """
    Load database credentials from .pgpass file
    
    Arguments:
        - pgpass_path: path to pgpass file containing db credentials
        - db: the name of the database being connected to

    Returns:
        - host, port, user, password, db
    """

    # Load credentials from path
    with open(os.path.expanduser(path), 'r') as file:
        host, port, _, user, password = file.read().strip().split(':')
    
    return host, port, user, password, db


def get_connection(pgpass_path='~/.pgpass', db=DB):
    """
    Get an SQL connection to a database.

    Arguments:
        - pgpass_path: path to pgpass file containing db credentials
        - db: the name of the database being connected to

    Returns:
        - conn: an SQL connection
    """
    host, port, user, password, db = get_credentials(path=pgpass_path, db=db)
    db_url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    conn = sqlalchemy.create_engine(db_url).connect()
    return conn


def run_sql_from_file(conn, path, replace={}):
    """
    Run an SQL script from a file.

    Arguments:
        - conn: SQL connection to database
        - path: path to SQL file
        - replace: ???
    """
    with open(path, 'r') as f:
        query = [s.strip() + ';' for s in f.read().split(';')[:-1]]
        for s in query:
            for k, v in replace.items():
                s = s.replace(k, v)
            run_sql_from_string(conn, s)


def run_sql_from_string(conn, statement):
    """
    Run SQL statement from a string.

    Arguments:
        - conn: SQL connection to database
        - statement: SQL statement being run
    """
    statement = sqlalchemy.text(statement)
    conn.execute(statement)


def get_table_columns(conn, table_name):
    """
    Get the column names from a table.

    Arguments:
        - conn: SQL connection to database
        - table_name: name of table
    """
    database_name = get_credentials()[-1]
    table_schema = table_name.split('.')[0]
    table_name = '.'.join(table_name.split('.')[1:])
    table_columns = pd.read_sql("select column_name from information_schema.columns "
                                f"where table_catalog = '{database_name}' "
                                f"and table_schema = '{table_schema}' "
                                f"and table_name = '{table_name}';", conn)
    return table_columns['column_name'].tolist()

