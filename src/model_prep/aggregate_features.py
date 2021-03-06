import numpy as np
from ..utils import sql_utils as sql



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

    # check whether to create an imputation flag
    impute_col_flag = not imputation.endswith('_noflag')

    # determine type of imputation and create appropriate sql code to do so
    if imputation.startswith('zero'):
        impute_sql = '0'
    elif imputation.startswith('inf'):
        impute_sql = '100000000'
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
        - conn: connection to the database
        - config: config file
        - cohort_table: str sql tablename for cohort to aggregate features from
        - to_table: str sql tablename for aggregated feature table to be stored
        - start_time: str in format 'YYYY-MM-DD' indicating the feature start time for the cohort
        - end_time: str in format 'YYYY-MM-DD' indicating the feature end time for the cohort
        - preprocessing_prefix: str indicating prefix to be used in preprocessng (if any)
    """
    join_query = f'select * from {cohort_table}'
    imputes = []

    # create a function to generate self-incrementing aliases
    t = 0
    def alias_counter():
        val = [0]
        def inc():
            val[0] += 1
            return f'alias_{val[0]}'
        return inc
    get_alias = alias_counter()

    # for each table features need to be aggregated from, impute missing features, and create sql string that aggregates features by facility, and left joins them to the cohort table
    for table_ix, agg_table in enumerate(config):
        output_prefix = agg_table['prefix']
        input_table = agg_table['from_table']
        input_table = input_table.replace('{prefix}', preprocessing_prefix)
        table_type = agg_table['table_type']

        # if the row driver is facilities, impute features and join to the cohort table
        if table_type == 'entity':
            column_avoid_list = ['entity_id']
            if 'event_date_column_name' in agg_table:
                event_date_col_name = agg_table['event_date_column_name']
                column_avoid_list += [event_date_col_name]
            table_columns = sql.get_table_columns(conn, input_table)
            feature_names = [x for x in table_columns if not x in column_avoid_list]
            if 'columns' in agg_table:
                feature_names = [s for s in feature_names if s in agg_table['columns']]

            for feature_name in feature_names:
                imputation = agg_table['imputation']
                impute_config = get_impute_str(feature_name, imputation)
                imputes.append((feature_name,) + impute_config)

            if 'event_date_column_name' in agg_table:
                # filter event date and pick the latest record for each entity
                get_columns_query = f'select entity_id, {event_date_col_name}, ' + \
                                        f'{", ".join(feature_names)} from {input_table} ' + \
                                    'right join (' + \
                                        f'select entity_id, max({event_date_col_name}) {event_date_col_name} ' + \
                                        f'from {input_table} ' + \
                                        f"where {event_date_col_name} >= '{start_time}'::date " + \
                                        f"and {event_date_col_name} <= '{end_time}'::date " + \
                                        'group by entity_id' + \
                                    f') as {get_alias()} using (entity_id, {event_date_col_name})'
                get_columns_query = f'select entity_id, {", ".join(feature_names)} ' + \
                                    f'from ({get_columns_query}) as {get_alias()}'
            else:
                get_columns_query = f'select entity_id, {", ".join(feature_names)} from {input_table}'

            join_query += ' '
            join_query += f'left join ({get_columns_query}) in_table_{table_ix} using (entity_id)'

        # if the row driver is inspections, impute features, aggregate to the facility level, and join to the cohort table
        elif table_type == 'event':
            feature_columns = []
            event_date_col_name = agg_table['event_date_column_name']
            knowledge_date_col_name = agg_table['knowledge_date_column_name']
            for agg_column in agg_table['aggregates']:
                agg_column_name = agg_column['column_name']
                for metric in agg_column['metrics']:
                    if metric == "datediff":
                        feature_name = f'{output_prefix}_days_since_{agg_column_name}'
                        feature_str = f"min('{end_time}'::date - {agg_column_name}) as {feature_name}"

                        feature_columns.append(feature_str)

                        imputation = agg_table['imputation'][metric]
                        impute_config = get_impute_str(feature_name, imputation)
                        imputes.append((feature_name,) + impute_config)

                    else:
                        if not 'time_windows' in agg_column:
                            agg_column['time_windows'] = ['all']

                        for time_window in agg_column['time_windows']:
                            feature_name = f'{output_prefix}_{metric}_{agg_column_name}'
                            if time_window == 'all':
                                feature_expression = agg_column_name
                            else:
                                feature_expression = 'case when ' + \
                                                     f"{event_date_col_name} > '{end_time}'::date - interval '{time_window}'" + \
                                                     f' then {agg_column_name} else null end'
                                feature_name += '_' + time_window.replace(' ', '_')

                            feature_str = f'{metric}({feature_expression}) as {feature_name}'

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
                f"where {event_date_col_name} >= '{start_time}'::date " + \
                f"and {event_date_col_name} <= '{end_time}'::date " + \
                f"and {knowledge_date_col_name} <= '{end_time}'::date " + \
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

    sql.run_sql_from_string(conn, f'create table {to_table} as ({select_sql});')
