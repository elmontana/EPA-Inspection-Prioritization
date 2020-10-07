import pandas as pd
import sklearn.linear_model as sk
import yaml

from sqlalchemy import create_engine

import psycopg2
import io
import pickle
import os
from pathlib import Path

import utils.sql_utils as sql



def load_grid_config():
	with open("grid_config.yaml", 'r') as stream:
	    data_loaded = yaml.safe_load(stream)
	#print(data_loaded)

	model_list = []
	for model_name, model_params in data_loaded["grid_config"].items():
		model_list.append((model_name, model_params))
		#print(model)
	return model_list


def get_df(feature_table, label_table):
	conn = sql.get_connection()
	sql_query = f'select f.*, l.label from {feature_table} f inner join {label_table} l on f.entity_id = l.entity_id;'
	dataframe = pd.read_sql(sql_query, con=conn)
	return dataframe

def train_model(feature_table, label_table, grid_config, model):
	df = get_df(feature_table, label_table)
	arr = df.to_numpy(copy=True)
	# import pdb
	# pdb.set_trace()

	X = arr[:, :-1]
	y = arr[:, -1]
	model.fit(X, y)
	
	return model

def save_model(model, log_dir, model_name):

	model_name += ".pkl"

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	root = Path(".")
	my_path = root / log_dir / model_name

	file_handler = open(my_path, 'wb')
	pickle.dump(model, file_handler)

def main(feature_table, label_table, grid_config, log_dir):
	#trained_model_list = []

	for model_name, model_params in grid_config:
		model = getattr(sk, model_name)(**model_params)
		print(model)
		trained_model = train_model(feature_table, label_table, grid_config, model)
		#trained_model_list.append(trained_model)

		save_model(trained_model, log_dir, model_name)

	return



if __name__ == '__main__':
	#conn = create_engine("postgresql://dhruvm:@mlpolicylab.db.dssg.io/epa3_database", connect_args={'options': '-csearch_path={}'.format("semantic")})
	main("semantic.reporting", "semantic.labels", load_grid_config(), "log_dir")


