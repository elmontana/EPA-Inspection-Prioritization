def num_labeled_samples(y, y_pred): 
    """
    Get the number of labeled samples for a given set of model predictions

    Arguments:
        - y: label array
        - y_pred: array of label predictions for which ground truth is known

    Returns:
        - num_labeled_samples: number of labeled samples for a given set of model predictions
    """
    return y_pred.sum()
