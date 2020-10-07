import os
import pickle
import sklearn.linear_model as sk
import yaml

from pathlib import Path
from utils.data_utils import get_data



def get_models(config_path='./grid_config.yaml'):
    """
    Create models as specificed by a config file.

    Arguments:
        - config_path: path to configuration file

    Returns:
        - models: a list of models
    """
    with open('grid_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    models = []
    for model_name, model_kwargs in config['grid_config'].items():
        model = getattr(sk, model_name)(**model_kwargs)
        models.append(model)

    return models



def train(feature_table, label_table, config_path='./grid_config.yaml', save_dir='./saved_models'):
    """
    Train models as specified by a config file.

    Arguments:
        - feature_table: name of table containing test features
        - label_table: name of table containing label features
        - config_path: path to configuration file
        - save_dir: directory for saving models
    """

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data and models
    X, y = get_data(feature_table, label_table)
    models = get_models(config_path)

    # Train models
    for model in models:
        model.fit(X, y)

        # Save model
        model_path = Path(save_dir) / f'{model.__class__.__name__}.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)



if __name__ == '__main__':
    train('semantic.reporting', 'semantic.labels')

