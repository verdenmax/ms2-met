import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from joblib import dump
from .base_model import BaseModel


class SklearnLRModel(BaseModel):
    def __init__(self, model_params, training_params, feature_names):
        super().__init__(feature_names)
        # 合并 params（sklearn 不需要 training_params，但保留接口）
        if training_params.get('valid_size', 0.0) > 0:
            logging.warning(
                "sklearn 训练过程不需要验证集，但是参数设置中验证集划分非0 %s",
                training_params.get('valid_size', 0.0))
        self.model_params = model_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = LogisticRegression(**self.model_params)
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)

    def _raw_feature_importance(self, importance_type='gain'):
        if self.model is None:
            return None
        # 使用系数绝对值作为“重要性”
        coefs = np.abs(self.model.coef_[0])
        return dict(zip(self.feature_names, coefs))
