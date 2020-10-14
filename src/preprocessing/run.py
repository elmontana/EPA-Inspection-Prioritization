import os
import glob
import sqlalchemy

from ..utils.sql_utils import run_sql_from_file



dir_path = os.path.dirname(os.path.realpath(__file__))
get_path = lambda s: os.path.join(dir_path, s)



def run_data_loading(conn, mode, prefix):
    if mode == 'acs':
        print('ACS data loading not implemented yet.')


def run_step(conn, sql_files, prefix):
    for sql_filename in sql_files:
        run_sql_from_file(
            conn,
            os.path.join(get_path(f'sql/{sql_filename}')),
            replace={'{prefix}': prefix})


def main(conn, config):
    prefix = config['prefix']
    run_data_loading(conn, 'acs', config['prefix'])
    for sql_group in config['sql']:
        run_step(conn, sql_group['files'], config['prefix'])



if __name__ == '__main__':
    main()
