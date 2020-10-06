import os
import pandas as pd
import sqlalchemy


def get_connection():
    # get db config
    host = os.environ['POSTGRES_HOST']
    user = os.environ['POSTGRES_USER']
    db = os.environ['POSTGRES_DB']
    password = os.environ['POSTGRES_PASSWORD']
    port = os.environ['POSTGRES_PORT']

    # create db connection
    db_url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    db_engine = sqlalchemy.create_engine(db_url)
    conn = db_engine.connect()

    return conn


def run_sql_from_file(conn, filename):
    print(f'Running {filename}...', end=' ')
    with open(filename, 'r') as f:
        query = [s.strip() + ';' for s in f.read().split(';')[:-1]]
        for s in query:
            run_sql_from_string(conn, s)
    print('done.')


def run_sql_from_string(conn, s):
    statement = sqlalchemy.text(s)
    conn.execute(statement)


def get_table_columns(conn, table_name):
    database_name = os.environ['POSTGRES_DB']
    table_schema = table_name.split('.')[0]
    table_name = '.'.join(table_name.split('.')[1:])
    table_columns = pd.read_sql("select column_name from information_schema.columns "
                                f"where table_catalog = '{database_name}' "
                                f"and table_schema = '{table_schema}' "
                                f"and table_name = '{table_name}';", conn)
    return table_columns['column_name'].tolist()
