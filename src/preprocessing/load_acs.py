# -*- coding: utf-8 -*-
"""
Data Collection and ETL

Created on Mon 11/02/2020 2020

@author: Erika Montana
"""
import censusdata
import re
from sqlalchemy import create_engine

### editable section
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
table_name = "acs_data"
survey = 'acs5'
year = 2013

### End editable section

##Part 1 - get data from Census
counties = censusdata.geographies(censusdata.censusgeo([('state','36'), ('county','*')]),survey,year , key='db8c95da0a4bf1d0f0b43c6e66158daaef578790')
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
engine = create_engine("postgresql://dhruvm:@mlpolicylab.db.dssg.io/epa3_database", connect_args={'options': '-csearch_path={}'.format("data_exploration")})
data.pg_copy_to(table_name, engine, if_exists='replace',index=False)