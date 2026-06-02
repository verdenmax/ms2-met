import logging
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from .base_model import BaseModel


class SklearnRFModel(BaseModel):
    """ Sklearn 随机森林 """

    def __init__(self, model_params, training_params, feature_names):
        super().__init__(feature_names)
        # 合并 params（sklearn 不需要 training_params，但保留接口）
        if training_params.get('valid_size', 0.0) > 0:
            logging.warn(f"sklearn 训练过程不需要验证集，但是参数设置中验证集划分非0 {
                         training_params.get('valid_size', 0.0)}")
        self.model_params = model_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Sklearn 不支持早停，忽略 val（或可用于手动早停，此处简化）
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # 返回正类概率

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)

    def _raw_feature_importance(self, importance_type='gain'):
        # Sklearn 只有一种 importance（基于基尼或信息增益）
        if self.model is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))
