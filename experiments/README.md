# Experiments

Each experiment is specified by a configuration file (see an example in [`test_run.yaml`](https://github.com/dssg/mlpolicylab_fall20_epa3/blob/master/experiments/test_run.yaml)). 
Here we provide some general information about creating and understanding these experiment config files.

#### `temporal_config`
There are 5 important fields within `temporal_config`. These are:
- `feature_start_time`: the start date from which we are taking features (in the form 'yyyy-mm-dd')
- `feature_duration`: duration over which to aggregate features (e.g. '2 years')
- `label_duration`: duration over which to aggregate labels (e.g. '1 year'). We assume that the label time period starts exactly at the end of the feature time period.
- `num_train_repeat`: 
- `train_repeat_interval`: 

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
The `(max_depth, max_features)` parameters for each model would be: `(1, None)`, `(1, 'sqrt')`, `(5, None)`, and `(5, 'sqrt')`.
