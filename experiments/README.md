# Experiments

Each experiment is specified by a configuration file (e.g. [`test_run.yaml`](experiments/test_run.yaml)). 
Here we provide some general information about creating experiment configs.

#### `temporal_config`

#### `cohort_config`

#### `preproccessing_config`

#### `feature_config`

#### `label_config`

#### `model_config`

The `model_config` field specifies configurations for each model to be trained in this experiment.
Each entry contains the name of a model class (e.g. "sklearn.tree.DecisionTreeClassifier"). 
Within each entry, we can specify a set of keyword arguments and values to pass to the model class constructor.

One important thing to note is that each keyword argument expects a list of values. Models are then created for every combination of values. 
For example, the following entry would create 4 separate decision-tree models.
```
'sklearn.tree.DecisionTreeClassifier':
    max_depth: [1, 5]
    max_features: [null, 'sqrt']
```
The `(max_depth, max_features)` parameters for each model would be: `(1, null)`, `(1, 'sqrt')`, `(5, null)`, and `(5, 'sqrt')`.
