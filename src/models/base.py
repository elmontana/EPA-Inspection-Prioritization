from abc import ABC, abstractmethod



class BaseModel(ABC):

    @abstractmethod
    def fit(X, y):
        raise NotImplementedError


    @abstractmethod
    def predict(X, columns=None):
        raise NotImplementedError


    @abstractmethod
    def predict_proba(X, columns=None):
        raise NotImplementedError


    def feature_importance(self, *args, **kwargs):
        raise NotImplementedError
