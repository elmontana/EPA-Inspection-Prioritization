import numpy as np

from src.models.base_model import BaseModel


class CommonSenseBaseline(BaseModel):
    def __init__(self, column=None, sort='desc'):
        assert column is not None
        self.column = column

        assert sort in ['asc', 'desc']
        self.sort = sort

    def fit(self, X, y, columns=None):
        pass

    def predict(self, X, columns=None):
        probs = self.predict_proba(X, columns)[:, 1]
        pred = np.zeros_like(probs)
        pred[probs >= 0.5] = 1.0
        return pred

    def predict_proba(self, X, columns=None):
        # sanity check
        assert columns is not None, \
            'Commonsense baseline requires column names.'
        assert self.column in columns, \
            f'Column name {self.column} not in list of columns'

        # sort data by column
        order = np.argsort(X[:, columns.index(self.column)]).flatten()
        if self.sort == 'asc':
            order = order[::-1]

        # calculate probabilities
        probs = np.zeros_like(order).astype(float)
        probs[order] = np.arange(len(order))
        probs /= np.max(probs)

        # convert result to two columns
        probs = np.stack([1.0 - probs, probs], axis=-1)

        return probs
