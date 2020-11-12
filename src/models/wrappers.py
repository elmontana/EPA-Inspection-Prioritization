from src.models.base_model import BaseModel



class SKLearnWrapper(BaseModel):
    """
    Wrapper around sklearn classifiers
    """

    def __init__(self, model):
        self.model = model


    def fit(self, X, y, *args, **kwargs):
        if self.model.__module__ == 'sklearn.linear_model.LogisticRegression':
            pass # TODO: normalize X & y, and save normalization parameters

        return self.model.fit(X, y, *args, **kwargs)


    def predict(self, X, columns=None):
        if self.model.__module__ == 'sklearn.linear_model.LogisticRegression':
            pass # TODO: normalize X, using the same normalization parameters calculated in self.fit()

        return self.model.predict(X)


    def predict_proba(self, X, columns=None):
        if self.model.__module__ == 'sklearn.linear_model.LogisticRegression':
            pass # TODO: normalize X, using the same normalization parameters calculated in self.fit()
            
        return self.model.predict_proba(X)
