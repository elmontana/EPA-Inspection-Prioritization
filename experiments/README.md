# Experiment Configurations

Each experiment is specified by a configuration file (see an example in [`test_run.yaml`](https://github.com/dssg/mlpolicylab_fall20_epa3/blob/master/experiments/test_run.yaml)).
Here we provide some general information about creating and understanding these experiment config files.

#### `temporal_config`

There are 5 important fields within `temporal_config`. These are:
- `feature_start_time`: the start date from which we are taking features (in the form 'yyyy-mm-dd').
- `feature_duration`: duration over which to aggregate features (e.g. '2 years').
- `label_duration`: duration over which to aggregate labels (e.g. '1 year'). We assume that the label time period starts exactly at the end of the feature time period.
- `num_train_repeat`: the number of train-test set splits to generate.
- `train_repeat_interval`: the interval for building multiple train-test set splits.

Below, we show an illustration of what the temporal config parameters mean.

![Illustration of Temporal Config](https://github.com/dssg/mlpolicylab_fall20_epa3/blob/master/experiments/temporal_config_illustration.png)

#### `cohort_config`

The `cohort_config` contains one field, which is a query to obtain entities (or facilities in the EPA context) for the train-test split. The query need to contain an `{as_of_date}` placeholder for the last date of the split and a `{prefix}` placeholder for the preprocessed data table prefixes.

#### `preproccessing_config`

The preprocessing configuration contains two fields. The `prefix` field holds a string that will be prepended to the name of every preprocessed tables. This is done so that experiments with different preprocessing configurations can be executed at the same time.

The `sql` field holds a list of preprocessing steps. Each preprocessing step is a dictionary with two keys: `name`, which is the name of the preprocessing step (this is not used in the program and exists for readability purposes only); `files`, which is a list of SQL files that should be executed in the step.

When preprocessing is run, the list of steps will be executed in order, and each SQL file in a step will be executed in order.

#### `feature_config`

The feature configuration contains all the information for all labels to be collected from all source tables. The outer structure of the configuration is a list of data source feature configs. Each item in the list defines features, aggregations, and imputations to be performed for one data source (ie. table).

In each data source feature configuration, the following fields are present:
- `prefix`: the prefix of all feature names obtained from this data source.
- `from_table`: the name of the table where the data will be fetched and aggregated. The table names can contain a `{prefix}` placeholder for preprocessed table name prefixes.
- `table_type`: this field must be either `entity` or `event`. If table type is `entity`, then the table source should contain per-entity records and no aggregation should be necessary (under which condition all features in the table are taken and the `aggregation` field is ignored); if table type is `event`, then the table source should contain per-event records and aggregation will be necessary.
- `imputation`: the type of imputation to be done for the features from this source. If the table type is `entity`, then a single value is set for this field; if the table type is `event`, then this field should be a dictionary defining an imputation method for every aggregation method. Imputation methods can be `zero`, `zero_noflag`, `mean`, `mean_noflag`, `min`, `min_noflag`, `max`, and `max_noflag`. If the imputation method has a `noflag` suffix, then no imputation flag column will be created; otherwise a column identifying whether the feature is imputed will be created.
- `aggregates`: a list containing columns and aggregation methods. Each list item should contain a dictionary with field `column_name` specifying feature column and `metrics` containing a list of aggregation metrics (possible values: `count`, `sum`, `mean`) that should be used when creating aggregated features. If multiple metrics are specified, multiple columns of features will be created, one for each metric.

#### `label_config`

The label configuration contains a single query that selects the labels for a train-test split. The query should contain a `{start_date}` and an `{end_date}` prefix for filling in actual label selection start and end dates. When selecting from preprocessed tables, the prefix placeholder `{prefix}` should be used.

The query in the label config should output a table that contains two columns: a column named `entity_id` that contains unique entity identifiers, and a column named `label` that contains labels for the entities. The query does not need to perform filtering on cohort, as an inner join between the label table and the cohort table will be done later.

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

#### `eval_config`

The `eval_config` field specifies the metrics to use for evaluating the models. The `metrics` item defines the set of metrics to use, while the `k` item defines a list of top-k% or top-n values that the metrics will be evaluated on.
