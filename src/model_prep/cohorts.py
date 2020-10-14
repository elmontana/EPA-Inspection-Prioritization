from .aggregate_features import main as aggregate_features
from .select_labels import main as select_labels

from ..utils import sql_utils as sql
from ..utils.date_utils import date_to_string



def generate_cohort_table(conn, cohort_config, as_of_date, in_prefix, out_prefix):
    """
    Creates a table of facility data with information from before the passed in date.
    
    Arguments:
        - conn: a connection to the database
        - cohort_config: dictionary with cohort configurations
        as_of_date: str in format 'YYYY-MM-DD' indicating most recent date in the new table
        in_prefix: str providing prefix for table from which to select data
        out_prefix: str providing prefix for the table created
        
    Returns:
        cohort_table_name: str name of the table created
    """
    cohort_table_name = f'{out_prefix}_cohort'
    cohort_sql = cohort_config['query'].replace('{as_of_date}', as_of_date) \
                                       .replace('{prefix}', in_prefix)
    drop_sql = f'drop table if exists {cohort_table_name};'
    create_sql = f'create table {cohort_table_name} as ({cohort_sql});'
    sql.run_sql_from_string(conn, drop_sql)
    sql.run_sql_from_string(conn, create_sql)
    return cohort_table_name


def prepare_cohort(config, train_dates, test_dates, preprocessing_prefix, exp_table_prefix):
    """
    Generate a cohort, then create tables for features & labels, 
    within the given train and test date ranges.

    Ar
    """

    # Get database connection
    conn = sql.get_connection()

    # Generate cohort table
    cohort_as_of_date = date_to_string(test_dates['label_end_time'])
    cohort_table_name = generate_cohort_table(
        conn,
        config['cohort_config'],
        cohort_as_of_date,
        preprocessing_prefix,
        exp_table_prefix)

    # Aggregate features for train & test data into new tables
    train_feature_table_name = f'{exp_table_prefix}_train_features'
    aggregate_features(
        conn, config['feature_config'], 
        cohort_table_name,
        train_feature_table_name,
        date_to_string(train_dates['feature_start_time']),
        date_to_string(train_dates['feature_end_time']),
        preprocessing_prefix)
    test_feature_table_name = f'{exp_table_prefix}_test_features'
    aggregate_features(
        conn, config['feature_config'], 
        cohort_table_name,
        test_feature_table_name,
        date_to_string(test_dates['feature_start_time']),
        date_to_string(test_dates['feature_end_time']),
        preprocessing_prefix)

    # Create new tables containing labels for each train & test data instance
    train_label_table_name = f'{exp_table_prefix}_train_labels'
    select_labels(
        conn, config['label_config'],
        train_label_table_name,
        date_to_string(train_dates['label_start_time']),
        date_to_string(train_dates['label_end_time']),
        preprocessing_prefix)
    test_label_table_name = f'{exp_table_prefix}_test_labels'
    select_labels(
        conn, config['label_config'], 
        test_label_table_name,
        date_to_string(test_dates['label_start_time']),
        date_to_string(test_dates['label_end_time']),
        preprocessing_prefix)

    return (train_feature_table_name, train_label_table_name, 
        test_feature_table_name, test_label_table_name)
