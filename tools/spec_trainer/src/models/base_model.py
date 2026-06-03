from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names  # 👈 新增：记录特征名

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

    @abstractmethod
    def _raw_feature_importance(self, importance_type='gain'):
        """返回原始重要性，格式由子类决定"""
        pass

    def feature_importance(self, importance_type='gain'):
        """
        返回与 self.feature_names 对齐的重要性数组
        长度 = len(self.feature_names)，顺序严格一致
        """
        if self.feature_names is None:
            raise ValueError("feature_names must be set to align importance!")

        raw_imp = self._raw_feature_importance(importance_type)

        if raw_imp is None:
            return np.zeros(len(self.feature_names))

        # 情况1: raw_imp 是 dict {feature_name: imp}
        if isinstance(raw_imp, dict):
            return np.array([
                raw_imp.get(name, 0.0) for name in self.feature_names
            ])

        # 情况2: raw_imp 是 list/array，假设顺序与 self.feature_names 一致
        # （仅当训练时 X 的列顺序 == self.feature_names 时成立）
        elif len(raw_imp) == len(self.feature_names):
            return np.array(raw_imp)

        else:
            raise ValueError(
                f"Cannot align feature importance: "
                f"expected {len(self.feature_names)} features, "
                f"got {len(raw_imp)}"
            )
