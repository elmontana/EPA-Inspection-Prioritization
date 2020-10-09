import numpy as np

from utils.sql_utils import run_sql_from_string, get_table_columns


def get_impute_str(column_name, imputation):
    """
    Creates string for sql imputation
    
    Args:
        - column_name: str name of column to be imputed
        - imputation: str indicating method of imputation
    
    Returns:
        - impute_sql: str sql imputation code
        - impute_col_flag: boolean if true an impuatation flag column will be created
    """
    impute_sql = ''
    #check whether to create an imputation flag
    impute_col_flag = not imputation.endswith('_noflag')
    
    #determine type of imputation and create appropriate sql code to do so
    if imputation.startswith('zero'):
        impute_sql = '0'
    elif imputation.startswith('mean') or imputation.startswith('avg'):
        impute_sql = f'avg({column_name}) over ()'
    elif imputation.startswith('min'):
        impute_sql = f'min({column_name}) over ()'
    elif imputation.startswith('max'):
        impute_sql = f'max({column_name}) over ()'
    else:
        raise ValueError(f'Unrecognized impute method {imputation}.')
    
    return (impute_sql, impute_col_flag)


def main(conn, config, cohort_table, to_table,
         start_time='0000-01-01', end_time='9999-12-31',
         preprocessing_prefix=None):
    """
    Aggregates Features
    
    Args:
        -conn: connection to the database
        -config: config file
        -cohort_table: str sql tablename for cohort to aggregate features from
        -to_table: str sql tablename for aggregated feature table to be stored
        -start_time: str in format 'YYYY-MM-DD' indicating the feature start time for the cohort
        -end_time: str in format 'YYYY-MM-DD' indicating the feature end time for the cohort
        -preprocessing_prefix: str indicating prefix to be used in preprocessng (if any)
    """
    join_query = f'select * from {cohort_table}'
    imputes = []
    # For each table features need to be aggregated from, impute missing features, and create sql string that aggregates features by facility, and left joins them to the cohort table
    for agg_table in config:
        output_prefix = agg_table['prefix']
        input_table = agg_table['from_table']
        input_table = input_table.replace('{prefix}', preprocessing_prefix)
        table_type = agg_table['table_type']
        
        #if the row driver is facilities, impute features and join to the cohort table
        if table_type == 'entity':
            table_columns = get_table_columns(conn, input_table)
            feature_names = [x for x in table_columns if x != 'entity_id']

            for feature_name in feature_names:
                imputation = agg_table['imputation']
                impute_config = get_impute_str(feature_name, imputation)
                imputes.append((feature_name,) + impute_config)

            join_query += ' '
            join_query += f'left join {input_table} using (entity_id)'
        
        # if the row driver is inspections, impute features, aggregate to the facility level, and join to the cohort table
        elif table_type == 'event':
            feature_columns = []
            date_col_name = agg_table['date_column_name']
            for agg_column in agg_table['aggregates']:
                agg_column_name = agg_column['column_name']
                for metric in agg_column['metrics']:
                    feature_name = f'{output_prefix}_{metric}_{agg_column_name}'
                    feature_str = f'{metric}({agg_column_name}) as {feature_name}'
                    feature_columns.append(feature_str)

                    imputation = agg_table['imputation'][metric]
                    impute_config = get_impute_str(feature_name, imputation)
                    imputes.append((feature_name,) + impute_config)

            join_query += ' '
            join_query += 'left join (' + \
                f'select ' + \
                    f'entity_id, ' + \
                    ', '.join(feature_columns) + ' ' + \
                f'from {input_table} ' + \
                f"where {date_col_name} >= '{start_time}'::date " + \
                f"and {date_col_name} <= '{end_time}'::date " + \
                f'group by entity_id order by entity_id' + \
            f') as {input_table.split(".")[-1]} using (entity_id)'

    # gather select elements for the final imputation
    select_columns = []
    for v in imputes:
        feature_str = f'coalesce({v[0]}, {v[1]}) ' + \
                      f'as {v[0]}'
        select_columns.append(feature_str)

        if v[2]:
            flag_str = f'(case when {v[0]} is null then 1 else 0 end) ' + \
                       f'as {v[0]}_impute_flag'
            select_columns.append(flag_str)
    select_str = ', '.join(select_columns)
    select_sql = f'select entity_id, {select_str} from ({join_query}) join_query'

    run_sql_from_string(conn, f'create table {to_table} as ({select_sql});')
