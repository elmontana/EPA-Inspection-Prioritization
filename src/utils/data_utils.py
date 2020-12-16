import pandas as pd
from .sql_utils import get_connection, get_table_names, get_table_columns



def get_data(feature_table, label_table, discard_columns=[]):
    """
    Get data from feature and label tables as pd.DataFrame objects.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - discard_columns: list of column names to discard

    Returns:
        - X: feature data frame
        - y: label data frame
    """

    # Query data from sql tables
    sql_query = f'select f.*, l.label from {feature_table} f left join {label_table} l on f.entity_id = l.entity_id;'
    df = pd.read_sql(sql_query, con=get_connection())

    # Process data
    df = df.set_index('entity_id')
    df = df.drop(columns=discard_columns)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y


def get_table(table_name, columns=['*']):
    """
    Get data from SQL table as a pd.DataFrame object.

    Arguments:
        - table_name: name of table
        - columns: list of columns to select

    Returns:
        - df: a data frame with the given table columns
    """
    column_string = ', '.join(columns)
    query = f'select {column_string} from {table_name}'
    df = pd.read_sql(query, con=get_connection())
    return df


def get_test_results_over_time(table_prefix):
    """
    Get data from test results over time for a single experiment run.

    Arguments:
        - table_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")

    Returns:
        - test_results: a list of pd.DataFrames, i.e. test results over time 
        - test_dates: list of test dates corresponding to test results
        - model_classes: a list of model classes (should be same across all result data frames)
    """

    # Get names of test result tables
    test_result_tables = get_table_names(
        get_connection(), 'results', prefix=table_prefix, suffix='test_results')

    # Get corresponding data frames
    test_results = [get_table(f'results.{table}') for table in test_result_tables]

    # Get test dates & sort results by date
    test_dates = [int(f'20{table.split("_")[-3][:2]}') for table in test_result_tables]
    test_dates, test_results = zip(*sorted(zip(test_dates, test_results)))

    # Get names of model classes from data frames
    model_classes = test_results[0]['model_class'].to_numpy(copy=True)
    model_classes = [model_class.rsplit('.', 1)[-1] for model_class in model_classes]

    return test_results, test_dates, model_classes



def get_baseline_model_idx(table_prefix):
    """
    Get model indices for baselines.

    Arguments:
        - table_prefix: prefix of train feature tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")

    Returns:
        - baseline_model_idx: the row indices of n the best models
    """

    # Get list of model classes
    test_result_tables = get_table_names(
        get_connection(), 'results', prefix=table_prefix, suffix='test_results')
    test_result_table = get_table(f'results.{test_result_tables[-1]}', columns=['model_class'])
    model_classes = test_result_table['model_class']

    # Return indices of `CommonSenseBaseline`
    return [
        i for i in range(len(model_classes))
        if model_classes[i].rsplit('.', 1)[-1] == 'CommonSenseBaseline']


def get_experiment_feature_names(table_prefix):
    """
    Get the names of features from a particular experiment run.

    Arguments:
        - table_prefix: prefix of train feature tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")

    Returns:
        - feature_names: a list of feature names
    """

    # Get names of test feature tables
    test_feature_tables = get_table_names(
        get_connection(), 'experiments', prefix=table_prefix, suffix='test_features')
    assert len(test_feature_tables) > 0

    # Get feature column names
    feature_names = get_table_columns(get_connection(), f'experiments.{test_feature_tables[0]}')
    feature_names = [name for name in feature_names if name not in {'split', 'entity_id'}]

    # Make names readable
    feature_dict = {
        'zip_population': 'Population (Zip)', 'county_population': 'Population (County)',
        'zip_density_sq_miles': 'Pop. Density (Zip)',
        'mean_county_income': 'Mean Household Income (County)',
        'events_sum_found_violation': '# Violations (Aggregated Since 2000)',
        'events_sum_found_violation_impute_flag': '# Violations (Missing)',
        'events_avg_found_violation': 'Percentage of Violations (Aggregated Since 2000)',
        'events_avg_found_violation_impute_flag': 'Percentage of Violations (Missing)',
        'events_sum_citizen_complaint': '# Citizen Complaints (Aggregated Since 2000)',
        'events_sum_citizen_complaint_impute_flag':  '# Citizen Complaints (Missing)',
        'events_sum_penalty_amount': 'Fine Amount (Aggregated Since 2000)',
        'events_sum_penalty_amount_impute_flag': 'Fine Amount (Missing)', 
        'events_days_since_event_date': 'Days Since Last Inspection', 
        'events_days_since_event_date_impute_flag': 'Days Since Last Inspection (Missing)',
        'd001': 'Ignitable Waste', 'd002': 'Corrosive Waste', 'd003': 'Reactive Waste',
        'd005': 'Barium', 'd007': 'Chromium', 'd008': 'Lead', 'd009': 'Mercury',
        'd011': 'Silver', 'd016': '2,4‐D (Herbicide)', 'd018': 'Benzene', 'd039': 'Tetrachloroethylene',
        'other_d': 'Toxic Waste (Other)', 
        'tcor_only': 'Toxic Waste (tcor_only)', 'tcor_icr': 'Toxic Waste (tcor_icr)',
        'icr': 'Waste (icr)', 'tcmt': 'Heavy Metals',
        'f001': 'f001 (Spent Solvents)', 'f002': 'f002 (Spent Solvents)', 
        'f003': 'fOO3 (Spent Solvents)', 'f005': 'fOO5 (Spent Solvents)',
        'f006': 'f006 (Electroplating Waste)', 'f039': 'Leachate Waste', 
        'f1_5': 'Spent Solvent Waste',
        'other_f': 'Industrial Waste (Other)',
        'p001': 'Chemical Waste (p001)', 
        'other_p': 'P-List Chemical Waste (Other)', 'other_u': 'U-List Chemical Waste (Other)',
        'other_k': 'K-List Industrial Waste (Other)', 'Waste (Other)'
        'labp': 'Lab Pack (Misc)',
        'events_sum_found_violation_1_year': '# Violations (Aggregated 1 Year)',
        'events_sum_found_violation_2_years': '# Violations (Aggregated 2 Years)',
        'events_sum_found_violation_5_years': '# Violations (Aggregated 5 Years)',
        'events_avg_found_violation_1_year': 'Percentage of Violations (Aggregated 1 Year)',
        'events_avg_found_violation_2_years': 'Percentage of Violations (Aggregated 2 Years)',
        'events_avg_found_violation_5_years': 'Percentage of Violations (Aggregated 5 Years)',
        'events_sum_citizen_complaint_1_year': '# Citizen Complaints (Aggregated 1 Year)',
        'events_sum_citizen_complaint_2_years': '# Citizen Complaints (Aggregated 2 Years)',
        'events_sum_citizen_complaint_5_years': '# Citizen Complaints (Aggregated 5 Years)',
        'events_sum_penalty_amount_1_year': 'Fine Amount (Aggregated 1 Year)',
        'events_sum_penalty_amount_2_years': 'Fine Amount (Aggregated 2 Years)',
        'events_sum_penalty_amount_5_years': 'Fine Amount (Aggregated 5 Years)',
    }
    
    feature_names = [feature_dict[name] if name in feature_dict else name for name in feature_names]
    return feature_names
