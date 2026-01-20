from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    def feature_importance(self, importance_type='gain'):
        # 默认返回 None，子类可重写
        return None
