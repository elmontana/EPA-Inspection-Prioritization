# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:12:50 2020

@author: Erika Montana
"""
from ..utils import sql_utils as sql



def main(conn, label_config, table_name, start_date, end_date,
         preprocessing_prefix):
    """
    Creates table in destination schema containing the primary key and the label for each observation

    Keyword Arguments:
    conn: connection to database
    label_config: config file
    table_name: name of resulting table
    start_date - string in 'YYYY-MM-DD' format indicating beginning of temporal group
    end_date - string in 'YYYY-MM-DD' format indicating end of temporal group
    preprocessing_prefix - prefix for the preprocessed tables
    """
    label_sql = label_config['query']
    label_sql = label_sql.replace('{prefix}', preprocessing_prefix)
    label_sql = label_sql.replace('{start_date}', start_date)
    label_sql = label_sql.replace('{end_date}', end_date)
    drop_sql = f'drop table if exists {table_name};'
    create_sql = f'create table {table_name} as ({label_sql});'
    sql.run_sql_from_string(conn, drop_sql)
    sql.run_sql_from_string(conn, create_sql)
