import src.utils.data_utils as data_utils
import src.utils.plot_utils as plot_utils



def plot_results_over_time(
    test_results_tables_prefix, 
    metrics=['precision_score_at_600'], base_rates=[0.02],
    figsize=(20, 10), save_dir='./plots/'):
    """
    Plot test results of provided metrics, over time.

    Arguments:
        - test_results_tables_prefix: prefix of test result tables
            (usually {user}_{version}_{exp_name}_{exp_time}, e.g. "i_v1_test_run_201113235700")
        - metrics: a list of metrics (str) to plot results for
        - base_rates: a list of base rates, one for each metric
        - figsize: the size of the plotted figure
        - save_dir: directory where plots should be saved
    """
    return plot_utils.plot_results_over_time(
        test_results_tables_prefix, 
        metrics=metrics, base_rates=base_rates, save_dir=save_dir)


def plot_metric_at_k(results_table_name, save_dir='./plots/'):
    """
    Arguments: 
        - results_table_name: name of results table
        - save_dir: directory where plots should be saved
    """
    results_table_name = f'results.{results_table_name.rsplit(".", 1)[-1]}'
    results_df = data_utils.get_table(results_table_name)
    raise NotImplementedError


def plot_pr_at_k(results_table_name, save_dir='./plots/'):
    """
    Arguments: 
        - results_table_name: name of results table
        - save_dir: directory where plots should be saved
    """
    results_df = data_utils.get_table(results_table_name)
    raise NotImplementedError



if __name__ == '__main__':
    # So that every time we want to plot something, 
    # we don't have to run main.py and spend an hour training models;
    # instead just use the results that are already in the database.
    
    # test_results_tables_prefix = 'i_v1_test_run_201113235700'
    # plot_results_over_time(test_results_tables_prefix)

    # results_table_name = 'i_v1_test_run_201113235700_120101_test_results'
    # plot_pr_at_k(results_table_name)
