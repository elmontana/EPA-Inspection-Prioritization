# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:12:50 2020

@author: Erika Montana
"""
from utils.sql_utils import run_sql_from_string

def main(conn, label_config, table_name, start_date, end_date):
    """
    Creates table in destination schema containing the primary key and the label for each observation

    Keyword Arguments:
    conn: connection to database
    label_config: config file
    table_name: name of resulting table
    start_date - string in 'YYYY-MM-DD' format indicating beginning of temporal group
    end_date - string in 'YYYY-MM-DD' format indicating end of temporal group   
    """
    label_sql = label_config['query'].replace('{start_date}', start_date).replace('{end_date}', end_date)
    drop_sql = f'drop table if exists {table_name};'
    create_sql = f'create table {table_name} as ({label_sql});'
    run_sql_from_string(conn, drop_sql)
    run_sql_from_string(conn, create_sql)
    

