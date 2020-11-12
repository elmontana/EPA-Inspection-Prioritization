from src.models.base_model import BaseModel
from sklearn.preprocessing import StandardScaler



class SKLearnWrapper(BaseModel):
    """
    Wrapper around sklearn classifiers
    """

    def __init__(self, model):
        self.model = model
        self.normalizer = StandardScaler()

        self.should_normalize_inputs = False
        if self.model.__class__.__name__ == 'LogisticRegression':
            self.should_normalize_inputs = True


    def fit(self, X, y, *args, **kwargs):
        if self.should_normalize_inputs:
            X_normalized = self.normalizer.fit_transform(X)
            self.model.fit(X_normalized, y, *args, **kwargs)

        return self.model.fit(X, y, *args, **kwargs)


    def predict(self, X, columns=None):
        if self.should_normalize_inputs:
            X_normalized = self.normalizer.transform(X)
            return self.model.predict(X_normalized)

        return self.model.predict(X)


    def predict_proba(self, X, columns=None):
        if self.model.__module__ == 'sklearn.linear_model.LogisticRegression':
            X_normalized = self.normalizer.transform(X)
            return self.model.predict_proba(X_normalized)

        return self.model.predict_proba(X)
