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
        xlabel = 'k' if x_value_type == 'float' else 'n'
        p_values = [float(row[p_prefix + s]) for s in p_xs]
        r_values = [float(row[r_prefix + s]) for s in r_xs]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Precision', color=color)
        ax1.plot(p_x, p_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Recall', color=color)
        ax2.plot(r_x, r_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0.0, 1.0)

        fig.tight_layout()
        plt.savefig(str(save_prefix) + f'_pr_at_k_model_{index}.jpg', dpi=300)
        plt.close(fig)


def plot_feature_importances(feature_names, feature_importance, save_dir):
    assert len(feature_names) == len(feature_importance)
    y_pos = np.arange(len(feature_names))
    order = np.argsort(feature_importance)[::-1]
    feature_importance = feature_importance[order]
    feature_names = [feature_names[order[i]] for i in range(len(order))]

    fig, ax = plt.subplots()
    ax.barh(y_pos, feature_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    fig.set_size_inches(11.0, 8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'feature_importance.pdf'))
    plt.close(fig)
