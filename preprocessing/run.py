import os
import glob
import sqlalchemy


dir_path = os.path.dirname(os.path.realpath(__file__))
get_path = lambda s: os.path.join(dir_path, s)


def run_sql(conn, filename):
    print(f'Running {filename}...', end=' ')
    with open(filename, 'r') as f:
        query = [s.strip() + ';' for s in f.read().split(';')[:-1]]
        for s in query:
            statement = sqlalchemy.text(s)
            conn.execute(statement)
    print('done.')


def run_data_loading(conn):
    pass


def run_step(conn, mode):
    run_sql(conn, get_path(f'sql/setup_{mode}.sql'))
    for filename in glob.glob(get_path(f'sql/{mode}_*.sql')):
        run_sql(conn, filename)


def main():
    # create db connection
    host = os.environ['POSTGRES_HOST']
    user = os.environ['POSTGRES_USER']
    db = os.environ['POSTGRES_DB']
    password = os.environ['POSTGRES_PASSWORD']
    port = os.environ['POSTGRES_PORT']

    db_url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    db_engine = sqlalchemy.create_engine(db_url)
    conn = db_engine.connect()

    run_data_loading(conn)
    run_step(conn, 'clean')
    run_step(conn, 'semantic')

    conn.close()


if __name__ == '__main__':
    main()
