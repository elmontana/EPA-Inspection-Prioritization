import click
import getpass
import glob
import os
import yaml

from attrdict import AttrDict
from model_prep.aggregate_features import main as aggregate_features
from model_prep.select_labels import main as select_labels
from preprocessing.run import main as run_preprocess
from utils.date_utils import parse_date, parse_interval
from utils.date_utils import date_to_string, get_current_time_string
from utils.sql_utils import get_connection, run_sql_from_string

from train import train
from evaluate import evaluate



def parse_temporal_config(temporal_config):
    """"
    Takes a config file and returns two lists of dictionaries for each iteration of model training/testing.
    One list is for the test set, the other is for the training set.
    Each dictionary contains critical feature and label start and endtimes.
    
    Args:
        temporal_config: config file with labels for feature_start_time, feature_duration, label_duration, and the train_repeat_interval
        
    Returns:
        train_splits: list of dictionaries of feature/label start/endtimes for various training sets
        test_splits: list of dictionaries of feature/label start/endtimes for various testing sets
    """"
    #Parse config file
    xs = parse_date(temporal_config['feature_start_time'])
    xi = parse_interval(temporal_config['feature_duration'])
    yi = parse_interval(temporal_config['label_duration'])
    ri = parse_interval(temporal_config['train_repeat_interval'])

    train_splits = []
    test_splits = []
    
    #For every training instance, create a dictionary of start and endtimes for the training and testing data
    for i in range(temporal_config['num_train_repeat']):
        train_xs = xs + ri * i 
        train_splits.append({
            'feature_start_time': train_xs,
            'feature_end_time': train_xs + xi,
            'label_start_time': train_xs + xi,
            'label_end_time': train_xs + xi + yi
        })

        test_xs = train_xs + yi ### I think this violates the disjoint test/training sets that Rayid was discussing yesterday, should add xi
        test_splits.append({
            'feature_start_time': test_xs,
            'feature_end_time': test_xs + xi,
            'label_start_time': test_xs + xi,
            'label_end_time': test_xs + xi + yi
        })

    return train_splits, test_splits


def gen_cohort_table(conn, cohort_config, as_of_date, in_prefix,
                          out_prefix):
    """
    Creates a table of facility data with information from before the passed in date.
    
    Args:
        conn: a connection to the database
        cohort_config: config file
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
    run_sql_from_string(conn, drop_sql)
    run_sql_from_string(conn, create_sql)
    return cohort_table_name


@click.command()
@click.option('--config', default='experiments/test_run.yaml',
              help='Path to config file.')
@click.option('--skip_preprocessing', is_flag=True,
              help='Whether to skip the preprocessing step.')
@click.option('--log_dir', type=str, default='logs',
              help='Directory to save trained model and testing results.')
def main(config, skip_preprocessing, log_dir):
    # get experiment config
    with open(config) as f:
        config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    # get db connection
    conn = get_connection()

    # get basic info of experiment
    exp_version = config['version']
    exp_name = config["experiment_name"]
    exp_time = get_current_time_string()[2:]
    username = getpass.getuser()[0]

    # preprocessing
    preprocessing_prefix = config['preprocessing_config']['prefix']
    if skip_preprocessing:
        print('Preprocessing skipped.')
    else:
        print('Preprosessing...')
        run_preprocess(conn, config['preprocessing_config'])
        print('Preprocessing done.')

    # load temporal config
    train_dates_list, test_dates_list = parse_temporal_config(config['temporal_config'])

    # training
    for train_dates, test_dates in zip(train_dates_list, test_dates_list):
        split_time_abbr = date_to_string(train_dates['feature_start_time'])
        split_time_abbr = split_time_abbr.replace('-', '')[2:]
        split_name = f'train_{split_time_abbr}'
        prefix = f'{username}_{exp_version}_{exp_name}_{exp_time}_{split_name}'
        exp_table_prefix = f'experiments.{prefix}'
        train_save_dir = os.path.join(os.getcwd(), log_dir, prefix,
                                      'train_' + exp_time)
        test_save_dir = os.path.join(os.getcwd(), log_dir, prefix,
                                     'test_' + exp_time)

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

        # Aggregate labels for train & test data into new tables
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

        # Train models as specified by config
        train(
            config, 
            train_feature_table_name, 
            train_label_table_name,
            save_dir=train_save_dir)

        # Evaluate our models on the training data
        model_paths = glob.glob(f'{train_save_dir}/*.pkl')
        train_results = evaluate(
            config, 
            train_feature_table_name,
            train_label_table_name,
            model_paths,
            log_dir=train_save_dir)

        # Evaluate our models on the test data
        test_results = evaluate(
            config, 
            test_feature_table_name,
            test_label_table_name,
            model_paths,
            log_dir=test_save_dir)



if __name__ == '__main__':
    main()
