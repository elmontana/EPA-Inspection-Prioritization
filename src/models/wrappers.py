from . import base
from sklearn.preprocessing import StandardScaler



class SKLearnWrapper(base.BaseModel):
    """
    Wrapper around sklearn classifiers.
    """

    def __init__(self, model):
        self.model = model
        self.normalizer = StandardScaler()

        self.should_normalize_inputs = False
        if self.model_type == 'LogisticRegression':
            self.should_normalize_inputs = True


    @property
    def model_type(self):
        return self.model.__class__.__name__


    def fit(self, X, y, *args, **kwargs):
        if self.should_normalize_inputs:
            X_normalized = self.normalizer.fit_transform(X)
            return self.model.fit(X_normalized, y, *args, **kwargs)

        return self.model.fit(X, y, *args, **kwargs)


    def predict(self, X, columns=None):
        if self.should_normalize_inputs:
            X_normalized = self.normalizer.transform(X)
            return self.model.predict(X_normalized)

        return self.model.predict(X)


    def predict_proba(self, X, columns=None):
        if self.should_normalize_inputs:
            X_normalized = self.normalizer.transform(X)
            return self.model.predict_proba(X_normalized)

        return self.model.predict_proba(X)


    def feature_importance(self, *args, **kwargs):
        if self.model_type == 'LogisticRegression':
            pass
        elif self.model_type == 'DecisionTreeClassifier':
            pass
        elif self.model_type == 'RandomForestClassifier':
            pass
        elif self.model_type == 'GradientBoostingClassifier':
            pass

        raise NotImplementedError
