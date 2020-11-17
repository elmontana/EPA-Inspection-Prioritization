import re
import numpy as np
import src.utils.plot_utils as utils

# def extract_year(model_path):
# 	x,y= re.split("^.*model_grid_\d*_", model_path)
# 	return "20"+y[0:2]

def test_best_model_selection():
	"""
	tests function below
	"""
	test_results_over_time = utils.get_test_results_over_time("i_v1_model_grid_201115015235")
	print(best_model_per_metric(test_results_over_time))
	



def best_model_per_metric(test_results_over_time):
	"""
	test_results_over_time: list that contains results for each of the years
	"""
	total_precision_score = np.zeros(len(test_results_over_time[0].index))
	max_min_precision_score = np.ones(len(test_results_over_time[0].index))


	for test_results in test_results_over_time
		total_precision_score += test_results[precision_score_at_600]
		max_min_precision_score = np.minimum(max_min_precision_score, test_results[precision_score_at_600])

	avg_precision_score = np.array(total_precision_score)/len(test_results_over_time)
	sorted_indices_avg = np.argsort(avg_precision_score) #note this is zero indexed and gives ascending order

	sorted_indices_max_min = np.argsort(max_min_precision_score)

	return [sorted_indices_avg[-1], sorted_indices_max_min[-1]]
