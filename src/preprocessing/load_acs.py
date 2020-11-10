# -*- coding: utf-8 -*-
"""
Data Collection and ETL

Created on Mon 11/02/2020 2020

@author: Erika Montana
"""
import censusdata
import re
from tqdm import tqdm
from sqlalchemy import create_engine

from ..utils.sql_utils import run_sql_from_string



def load_acs_data(conn, variables, variable_names, table_name, survey, year):
  """
  Downloads specified data about the state of New York from the ACS and creates a table in the database

  Arguments:
        - conn: db connection
        - variables: list of variable names to pull from ACS survey
        - variable_names: dictionary containing the variables as keys and the desired column names as values
        - table_name: name of table to create in db
        - survey: name of acs survey to pull data from
        - year: year of acs survey data should be pulled from
  """
  # drop table if already exists
  run_sql_from_string(conn, f'drop table if exists data_exploration.{table_name}')

  # get data from census
  census_geo = [('state','36'), ('county','*')]
  counties = censusdata.geographies(censusdata.censusgeo(census_geo),
                                    survey, year,
                                    key='db8c95da0a4bf1d0f0b43c6e66158daaef578790')
  countylist = list(counties.values())

  for county in tqdm(countylist, desc='Load ACS data'):
      params = county.params()
      if county == countylist[0]:
        data = censusdata.download(survey, year,
                             censusdata.censusgeo([params[0], params[1], ('block group', '*')]),
                             variables, key='db8c95da0a4bf1d0f0b43c6e66158daaef578790')
      else:
        data = data.append(censusdata.download(survey, year,
                             censusdata.censusgeo([params[0], params[1], ('block group', '*')]),
                             variables, key='db8c95da0a4bf1d0f0b43c6e66158daaef578790'))

  # transform data
  data.rename(columns=variable_names, inplace=True)
  data.reset_index(inplace=True)
  for i, col_name in enumerate(['state', 'county', 'tract', 'block group']):
      data[col_name] = data['index'].apply(lambda col: str(col.geo[i][1]))
  data = data.drop(['index'], axis=1)

  # load on database
  data.to_sql(table_name, conn, schema='data_exploration', index=False)
