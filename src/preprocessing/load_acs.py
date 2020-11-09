# -*- coding: utf-8 -*-
"""
Data Collection and ETL

Created on Mon 11/02/2020 2020

@author: Erika Montana
"""
import censusdata
import re
from sqlalchemy import create_engine

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
  ##Part 1 - get data from Census
  counties = censusdata.geographies(censusdata.censusgeo([('state','36'), ('county','*')]),survey, year, key='db8c95da0a4bf1d0f0b43c6e66158daaef578790')
  countylist = list(counties.values())

  for county in countylist:
      params = county.params()
      if(county==countylist[0]):
        data = censusdata.download(survey, year,
                             censusdata.censusgeo([params[0],params[1], ('block group', '*')]),
                             variables, key='db8c95da0a4bf1d0f0b43c6e66158daaef578790')
      else:
        data = data.append(censusdata.download(survey, year,
                             censusdata.censusgeo([params[0],params[1], ('block group', '*')]),
                             variables, key='db8c95da0a4bf1d0f0b43c6e66158daaef578790'))

  #Part 2 Transform data
  data.rename(columns=variable_names, inplace=True)
  data.reset_index(inplace=True)
  data['index'] = data['index'].apply(str)
  data['index'] = data['index'].apply(lambda x: re.sub(':.*','',x))
  data[['block_grp','tract','county', 'state']]=data['index'].str.split(',', expand=True)
  data['block_grp']=data['block_grp'].apply(lambda x: int(re.search('\d+',x).group(0)))
  data['tract']=data['tract'].apply(lambda x: int(re.search('\d+',x).group(0)))
  data = data.drop(['index'],axis=1)

  #Part 3 Load on Database
  data.to_sql(table_name, conn, schema='data_exploration')
