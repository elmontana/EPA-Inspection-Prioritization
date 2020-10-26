import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_metric_at_k(results, prefix, x_value_type='float', save_path=None):
    # clear figure
    plt.clf()

    # get x axis values from dataframe
    column_names = [s for s in list(results.columns) if s.startswith(prefix)]
    x_values_str = [s[len(prefix):] for s in column_names]
    if x_value_type == 'int':
        x_values_str = [s for s in x_values_str if not '.' in s]
        x_values = [int(s) for s in x_values_str]
    elif x_value_type == 'float':
        x_values_str = [s for s in x_values_str if '.' in s]
        x_values = [float(s) for s in x_values_str]
    else:
        raise ValueError('x_value type must be int or float.')
    
    # iterate models and plot graphs
    for index, row in results.iterrows():
        y_values = [float(row[prefix + s]) for s in x_values_str]
        plt.plot(x_values, y_values)

    # add axis labels and save figure
    xlabel = 'k' if x_value_type == 'float' else 'n'
    ylabel = f'{prefix}{xlabel}'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([f'Model {i}' for i in range(len(results))])
    plt.tight_layout()
    plt.savefig(save_path)
