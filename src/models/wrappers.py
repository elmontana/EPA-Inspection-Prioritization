from src.models.base_model import BaseModel



class SKLearnWrapper(BaseModel):
    """
    Wrapper around sklearn classifiers
    """

    def __init__(self, model):
        self.model = model


    def fit(self, X, y, *args, **kwargs):
        return self.model.fit(X, y, *args, **kwargs)


    def predict(self, X, columns=None):
        return self.model.predict(X)


    def predict_proba(self, X, columns=None):
        return self.model.predict_proba(X)
