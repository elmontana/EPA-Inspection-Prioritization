import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_x_axis_values(columns, prefix, x_value_type):
    column_names = [s for s in list(columns) if s.startswith(prefix)]
    x_values_str = [s[len(prefix):] for s in column_names]
    if x_value_type == 'int':
        x_values_str = [s for s in x_values_str if not '.' in s]
        x_values = [int(s) for s in x_values_str]
    elif x_value_type == 'float':
        x_values_str = [s for s in x_values_str if '.' in s]
        x_values = [float(s) for s in x_values_str]
    else:
        raise ValueError('x_value type must be int or float.')
    return x_values_str, x_values


def plot_metric_at_k(results, prefix, x_value_type='float', save_path=None):
    # clear figure
    plt.clf()

    # get x axis values from dataframe
    x_values_str, x_values = get_x_axis_values(results.columns, prefix,
                                               x_value_type)

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


def plot_pr_at_k(results, x_value_type, p_prefix, r_prefix, save_prefix):
    # get x axis values from dataframe
    p_xs, p_x = get_x_axis_values(results.columns, p_prefix, x_value_type)
    r_xs, r_x = get_x_axis_values(results.columns, r_prefix, x_value_type)

    for index, row in results.iterrows():
        plt.clf()
        p_values = [float(row[p_prefix + s]) for s in p_xs]
        r_values = [float(row[r_prefix + s]) for s in r_xs]
        plt.plot(p_x, p_values)
        plt.plot(r_x, r_values)
        xlabel = 'k' if x_value_type == 'float' else 'n'
        plt.legend(['Precision', 'Recall'])
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_pr_at_k_model_{index}.pdf')
