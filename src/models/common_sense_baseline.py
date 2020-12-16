import numpy as np
from . import base



class CommonSenseBaseline(base.BaseModel):
    """
    Implement common sense baselines 
    that rank samples based on a single feature column.
    """

    def __init__(self, column, sort='desc'):
        """
        Arguments:
            - column: the name of the feature column to use
            - sort: one of {'asc', 'desc'};
                specifies the order in which we sort our column values
        """
        assert sort in {'asc', 'desc'}
        self.column, self.sort = column, sort


    def fit(self, X, y):
        pass


    def predict(self, X, columns=None):
        probs = self.predict_proba(X, columns)[:, 1]
        return probs > 0.5


    def predict_proba(self, X, columns=None):
        # Sanity check
        assert columns is not None, 'CommonSenseBaseline requires a list of column names.'
        assert self.column in columns, f'"{self.column}" not in list of columns'

        # Sort data by column
        order = X[:, columns.index(self.column)].argsort().flatten()
        if self.sort == 'asc':
            order = order[::-1]

        # Assign probabilities based on order
        probs = np.zeros_like(order).astype(float)
        probs[order] = np.linspace(0, 1, num=len(order))

        # Convert result to two columns
        probs = np.stack([1.0 - probs, probs], axis=-1)

        return probs


    def feature_importance(self, columns=None):
        # Sanity check
        assert columns is not None, 'CommonSenseBaseline requires a list of column names.'
        assert self.column in columns, f'"{self.column}" not in list of columns'

        # Return one-hot feature importance vector
        importances = np.zeros(len(columns))
        importances[list(columns).index(self.column)] = 1
        return importances

