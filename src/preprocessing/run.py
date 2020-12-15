import os
import glob
import sqlalchemy
import tqdm

from .load_acs import load_acs_data, load_density_data
from ..utils.sql_utils import run_sql_from_file



dir_path = os.path.dirname(os.path.realpath(__file__))
get_path = lambda s: os.path.join(dir_path, s)



def run_data_loading(conn, mode, prefix):
    """
    Load demographic data and create corresponding db tables.

    Arguments:
        - conn: an SQL connection
        - mode: one of {'acs'}
        - prefix: string prefix for the names of any created tables 
    """
    if mode == 'acs':
        # choose ACS variables to include and load them
        variables = [
            'B01003_001E','B02001_002E', 'B02001_003E', 'B02001_005E','B19049_001E',
            'B15003_002E','B15003_017E','B15003_022E','B15003_023E','B15003_025E',
        ]
        variable_names = {
            'B01003_001E':'total_pop',
            'B02001_002E':'white_pop', 
            'B02001_003E':'black_pop',
            'B02001_005E':'asian_pop',
            'B19049_001E':'median_income',
            'B15003_002E':'edu_no_schooling',
            'B15003_017E':'edu_hsd',
            'B15003_022E':'edu_bs',
            'B15003_023E':'edu_ms',
            'B15003_025E':'edu_phd'
        }
        table_name = f'{prefix}_acs_data'
        survey = 'acs5'
        year = 2013
        load_acs_data(conn, variables, variable_names, table_name, survey, year)

        # load population density data
        density_table_name = f'{prefix}_pop_density_data'
        load_density_data(conn, density_table_name)

    else:
        raise NotImplementedError


def run_step(conn, sql_files, prefix):
    """
    Preprocess data from the db by running the given SQL scripts.

    Arguments:
        - conn: an SQL connection
        - sql_files: list of SQL files to run
        - prefix: string prefix for the names of any created tables 
    """
    sql_loop = tqdm.tqdm([get_path(f'sql/{filename}') for filename in sql_files])
    for path in sql_loop:
        sql_loop.set_description(path)
        run_sql_from_file(conn, path, replace={'{prefix}': prefix})


def main(conn, config, run_data_upload=False):
    """
    Run data preprocessing.

    Arguments:
        - conn: an SQL connection
        - config: preprocessing config dictionary
        - run_data_upload: whether or not to run data upload to db before preprocessing
    """
    prefix = config['prefix']
    if run_data_upload:
        run_data_loading(conn, 'acs', config['prefix'])
    for sql_group in config['sql']:
        run_step(conn, sql_group['files'], config['prefix'])



if __name__ == '__main__':
    main()
