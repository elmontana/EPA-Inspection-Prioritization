import importlib
import itertools
import numpy as np
import os
import pickle
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool

from pathlib import Path
from ..utils.data_utils import get_data



def create_model(model_class_name, model_kwargs):
    """
    Dynamically instantiate a model given its name and arguments.

    Arguments:
        - model_class_name: the name of the model class (e.g. 'sklearn.tree.DecisionTreeClassifier')
        - model_kwargs: a dictionary of keyword arguments to pass to the model's constructor

    Returns:
        - model: an instantiated model
    """
    module_name, class_name = model_class_name.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    return model_class(**model_kwargs)


def get_model_configurations(config):
    """
    Get the set of all model configurations specified by the config. 
    For each model keyword argument, the config specifies a list of potential values.
    This function enumerates all possible combinations.

    Arguments:
        - config: a configuration dictionary for an experiment (loaded from yaml)

    Returns:
        - model_configurations: a list of configurations in the form (model_name, kwargs)
    """
    model_configurations = []
    for model_name, model_kwargs_set in config['model_config'].items():
        values_set = itertools.product(*model_kwargs_set.values())
        kwargs_set = [{k: v for k, v in zip(model_kwargs_set.keys(), values)} for values in values_set]

        for kwargs in kwargs_set:
            model_configurations.append((model_name, kwargs))

    return model_configurations


def train_single_model_single_arg(arg):
    """
    Train a single model with provided model specifications and data, using a
    single argument to fit the imap interface.

    Arguments:
        - arg: a tuple that includes arguments to a train_single_model call.
    """
    experiment_name, model_num, model_config, save_dir, X, y = arg
    return train_single_model(experiment_name, model_num, model_config, save_dir, X, y)


def train_single_model(experiment_name, model_num, model_config, save_dir, X, y):
    """
    Train a single model with provided model specifications and data.

    Arguments:
        - experiment_name: name of the experiment
        - model_num: index of the model
        - model_config: configuration of the model
        - save_dir: directory to save the model
        - X: features
        - y: labels
    """
    class_name, kwargs = model_config

    # Create & fit model
    try:
        model = create_model(class_name, kwargs)
        model.fit(X, y)
    except:
        return None
    
    # Save model
    model_path = Path(save_dir) / f'{experiment_name}_{class_name}_{model_num}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Create model description
    description = f'Model #{model_num}\nPath: {model_path}\nClass: {class_name}\nKeyword Args: {kwargs}'
    return model_num, description


def train(config, feature_table, label_table, discard_columns=[], save_dir='./saved_models/'):
    """
    Train models as specified by a config file.

    Arguments:
        - config: configuration dictionary for this experiment  (loaded from yaml)
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - discard_columns: names of columns to discard before building matrices
        - save_dir: directory for saving models
    """

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    X, y = get_data(feature_table, label_table, discard_columns=discard_columns)
    X, y = X.to_numpy(copy=True), y.to_numpy(copy=True).astype(int)

    # Filter out rows where a label does not exist
    labeled_indices = np.logical_or(y == 0, y == 1)
    X, y = X[labeled_indices], y[labeled_indices]

    # Train models
    experiment_name = config['experiment_name']
    model_configurations = get_model_configurations(config)
    num_models = len(model_configurations)
    model_descriptions = [None] * num_models
    pool = Pool(processes=5)
    for _ in tqdm(pool.imap(train_single_model_single_arg,
                            zip(repeat(experiment_name, num_models),
                                list(range(num_models)), model_configurations,
                                repeat(save_dir, num_models),
                                repeat(X, num_models), repeat(y, num_models))),
                  total=num_models, desc='Training models'):
        if _ is not None:
            model_num, model_description = _
            model_descriptions[model_num] = model_description
    pool.close()

    # Filter out trainings that failed
    success_train_indices = [i for i in range(num_models) if model_descriptions[i] is not None]
    model_configurations = [x for i, x in enumerate(model_configurations) if i in success_train_indices]
    model_descriptions = [x for x in model_descriptions if x is not None]

    # Log the model descriptions
    log_path = Path(save_dir) / f'{experiment_name}_info.txt'
    log_text = '\n\n'.join(model_descriptions)
    with open(log_path, 'w') as log_file:
        log_file.writelines(log_text)

    model_summary = []
    for model_config in model_configurations:
        summary_dict = {'model_name': model_config[0]}
        summary_dict.update(model_config[1])
        model_summary.append(summary_dict)

    return model_summary
