import sklearn.linear_model
import sklearn.preprocessing
from . import base



class SKLearnWrapper(base.BaseModel):
    """
    Wrapper around sklearn classifiers.
    """

    def __init__(self, model):
        self.model = model
        self.normalizer = sklearn.preprocessing.StandardScaler()

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
            return np.abs(self.model.coef_[0])
        elif self.model_type == 'DecisionTreeClassifier':
            return self.model.feature_importances_
        elif self.model_type == 'RandomForestClassifier':
            return self.model.feature_importances_
        elif self.model_type == 'GradientBoostingClassifier':
            return self.model.feature_importances_
        else:
            return None



class LogisticRegression(SKLearnWrapper):
    """
    Automatically decide which solver to use,
    based on the penalty parameter that is provided.
    """

    def __init__(self, *args, **kwargs):
        if 'penalty' in kwargs:
            if kwargs['penalty'] in {'l1', 'elasticnet'}:
                kwargs['solver'] = 'saga'

        model = sklearn.linear_model.LogisticRegression(*args, **kwargs)
        super().__init__(model)

