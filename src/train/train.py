import importlib
import itertools
import numpy as np
import os
import pickle
import tqdm

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
    model_configurations = get_model_configurations(config)
    model_descriptions = []

    training_loop = tqdm.tqdm(model_configurations)
    for model_num, (class_name, kwargs) in enumerate(training_loop):
        training_loop.set_description(f'Model #{model_num}: {class_name}')

        # Create & fit model
        try:
            model = create_model(class_name, kwargs)
            model.fit(X, y)
        except:
            continue
        
        # Save model
        experiment_name = config['experiment_name']
        model_path = Path(save_dir) / f'{experiment_name}_{class_name}_{model_num}.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        # Create model description
        description = f'Model #{model_num}\nPath: {model_path}\nClass: {class_name}\nKeyword Args: {kwargs}'
        model_descriptions.append(description)

    # Log the model descriptions
    experiment_name = config['experiment_name']
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
