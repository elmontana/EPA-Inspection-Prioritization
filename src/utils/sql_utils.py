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


def get_table_names(conn, schema, prefix='', suffix=''):
    """
    Get the names of tables from the specified schema.

    Arguments:
        - conn: SQL connection to database
        - schema: name of schema where tables are located
        - prefix: prefix of tables to include
        - suffix: suffix of tables to include

    Returns:
        - table_names: a list of table names
    """
    query = f"select table_name from information_schema.tables where table_schema = '{schema}' order by object_id"
    table_names = pd.read_sql(query, con=get_connection()).to_numpy(copy=True).flatten()
    table_names = [t for t in table_names if t.startswith(prefix) and table.endswith(suffix)]
    return table_names


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


def merge_tables(table_names, output_table_name):
    """
    Merge tables via 'UNION ALL', and drop the original tables.

    Arguments:
        - tables_names: list of table names to be merged
        - output_table_name: name of output table
    """
    conn = get_connection()
    
    select_queries = [
        f'select {i} as split, s{i}.* from {table_name} s{i}' 
        for i, table_name in enumerate(table_names)]
    query = f'create table {output_table_name} as ({" union all ".join(select_queries)});'
    sql.run_sql_from_string(conn, query)

    # Drop the original tables
    for table_name in table_names:
        sql.run_sql_from_string(conn, f'drop table {table_name}')

