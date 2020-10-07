import os
import glob
import sqlalchemy

from utils.sql_utils import run_sql_from_file


dir_path = os.path.dirname(os.path.realpath(__file__))
get_path = lambda s: os.path.join(dir_path, s)


def run_data_loading(conn):
    pass


def run_step(conn, mode):
    run_sql_from_file(conn, get_path(f'sql/setup_{mode}.sql'))
    for filename in glob.glob(get_path(f'sql/{mode}_*.sql')):
        run_sql_from_file(conn, filename)


def main(conn):
    run_data_loading(conn)
    run_step(conn, 'clean')
    run_step(conn, 'semantic')


if __name__ == '__main__':
    main()
