
# EPA Team 3

### Getting Started

Clone this repo on the server where the database is located:
```
git clone https://github.com/dssg/mlpolicylab_fall20_epa3
```
Install all required packages:
```
cd mlpolicylab_fall20_epa3
pip3 install -r requirements.txt
```

### Running the Code

To run an experiment from a config file:
```
python3 main.py --config [config_file]
```

Each experiment generates a unique prefix of the form `{user}_{version}_{exp_name}_{exp_time}`, e.g. "i_v1_test_run_201113235700". To generate plots from an experiment, just run the following:
```
python3 plot.py --exp [exp_prefix]
```

To get information about different command line arguments, run `python3 main.py --help` or `python3 plot.py --help`.


### How It Works

Every experiment is specified by a configuration file. Detailed documentation about configuration files can be found in [`experiments/README.md`](https://github.com/dssg/mlpolicylab_fall20_epa3/blob/master/experiments/README.md).

Running an experiment does the following:
1. **Setup and clean our data** with a [series of SQL scripts](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/preprocessing/sql) (only if the `--run_preprocessing` is included). Creates tables in the `cleaned` and `semantic` schemas of the database. These tables are for general use across all experiments, so `--run_preprocessing` only needs to be run once.
2. **Generate cohorts** of facilities, and **split the data** into training/validation sets. This involves aggregating features, imputing missing values, and computing labels. Creates tables in the `experiments` schema. These tables are specific to a particular experiment configuration.
3. **Train the different models** (across all specific model types and configurations). Performs grid search over the set of parameter combinations specified by the experimental config.
4. **Evaluate the models**, and save both the raw predictions and the evaluation metric results to a database. Creates tables in the `predictions` and `results` schemas.

### Code Structure

The code for this repository is currently structured as follows: 
* [`main.py`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/main.py): running experiments
* [`plot.py`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/plot.py): plotting and visualizing results
* [`experiments/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/experiments): experimental configuration files
* [`src/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src): source code files
	* [`src/preprocessing/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/preprocessing): preprocessing and cleaning raw data from EPA, NYSDEC, and ACS
	* [`src/model_prep/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/model_prep): generating cohorts, aggregating features, and computing labels
	* [`src/models/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/models): custom classes/wrappers/implementations for models
	* [`src/train/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/train): model training
	* [`src/evaluate/`](https://github.com/dssg/mlpolicylab_fall20_epa3/tree/master/src/evaluate): getting model predictions, model evaluation, and model selection
