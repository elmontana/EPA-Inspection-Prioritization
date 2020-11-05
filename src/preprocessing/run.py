import os
import glob
import sqlalchemy

from .load_acs import load_acs_data
from ..utils.sql_utils import run_sql_from_file



dir_path = os.path.dirname(os.path.realpath(__file__))
get_path = lambda s: os.path.join(dir_path, s)


def run_data_loading(conn, mode, prefix):
    if mode == 'acs':
        #choose ACS variables to include and load them
        variables = ['B01003_001E','B02001_002E', 'B02001_003E','B02001_005E','B19049_001E','B15003_002E','B15003_017E','B15003_022E','B15003_023E','B15003_025E']
        variable_names = {'B01003_001E':'total_pop',
                  'B02001_002E':'white_pop', 
                  'B02001_003E':'black_pop',
                  'B02001_005E':'asian_pop',
                  'B19049_001E':'median_income',
                  'B15003_002E':'edu_no_schooling',
                  'B15003_017E':'edu_hsd',
                  'B15003_022E':'edu_bs',
                  'B15003_023E':'edu_ms',
                  'B15003_025E':'edu_phd'}
        table_name = prefix + "_acs_data"
        survey = 'acs5'
        year = 2013
        load_acs_data(conn, variables, variable_names, table_name, survey, year)


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
