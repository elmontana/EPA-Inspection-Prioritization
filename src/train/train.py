import importlib
import itertools
import numpy as np
import os
import pickle
import tqdm

from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

from ..models.wrappers import SKLearnWrapper
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
    model = model_class(**model_kwargs)

    # Wrap sklearn models
    if model.__module__.startswith('sklearn'):
        model = SKLearnWrapper(model)

    return model


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


def train_single_model(experiment_name, model_index, model_config, save_dir, X, y):
    """
    Train a single model with provided model specifications and data.

    Arguments:
        - experiment_name: name of the experiment
        - model_index: index of the model
        - model_config: configuration of the model;
            a tuple of the form (class_name, kwargs)
        - save_dir: directory to save the model
        - X: feature array
        - y: label array

    Returns:
        - model_config: the configuration of the model
        - model_path: the path to the saved model
    """
    class_name, kwargs = model_config

    # Create & fit model
    try:
        model = create_model(class_name, kwargs)
        model.fit(X, y)
    except Exception as e:
        print(e)
        return model_config, None
    
    # Save model
    model_path = Path(save_dir) / f'{experiment_name}_{class_name}_{model_index}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return model_config, model_path


def train_single_model_unpack_args(args):
    """
    Train a single model with provided model specifications and data, 
    using a single argument to fit the imap interface.

    Arguments:
        - args: a tuple with the arguments to a `train_single_model` call.
    """
    return train_single_model(*args)


def train_multiprocessing(config, X, y, save_dir, num_processes=8):
    """
    Train models in parallel.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
        - X: feature array
        - y: label array
        - save_dir: directory for saving models
        - num_processes: number of different processes used for training

    Returns:
        - model_configurations: list of configurations of trained models
        - model_paths: list of paths to saved trained models
    """
    experiment_name = config['experiment_name']
    model_configurations = get_model_configurations(config)
    num_models = len(model_configurations)

    sucessful_model_configurations = []
    sucessful_model_paths = []

    pool = Pool(processes=num_processes)
    args = zip(
        repeat(experiment_name, num_models),
        range(num_models), 
        model_configurations,
        repeat(save_dir, num_models),
        repeat(X, num_models), 
        repeat(y, num_models))

    for model_config, model_path in tqdm.tqdm(
        pool.imap(train_single_model_unpack_args, args),
        total=num_models, desc='Training models'):

        if model_path is not None:
            sucessful_model_configurations.append(model_config)
            sucessful_model_paths.append(model_path)

    pool.close()
    return sucessful_model_configurations, sucessful_model_paths


def train(config, feature_table, label_table, discard_columns=[], save_dir='./saved_models/'):
    """
    Train models as specified by a config file.

    Arguments:
        - config: configuration dictionary for this experiment (loaded from yaml)
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
    model_configurations, model_paths = train_multiprocessing(config, X, y, save_dir)

    # Summarize models
    model_summaries = []
    for model_config, model_path in zip(model_configurations, model_paths):
        model_class, model_kwargs = model_config
        summary_dict = {
            'model_class': model_class,
            'model_path': str(model_path),
            **model_kwargs,
        }
        model_summaries.append(summary_dict)

    # Log model summaries
    log_path = Path(save_dir) / f'{config["experiment_name"]}_info.txt'
    log_text = '\n\n'.join([str(s) for s in model_summaries])
    with open(log_path, 'w') as log_file:
        log_file.writelines(log_text)

    return model_summaries
