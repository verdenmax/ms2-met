import lightgbm as lgb
import os
import numpy as np
from .base_model import BaseModel


class LGBModel(BaseModel):
    def __init__(self, model_params, training_params, feature_names):
        super().__init__(feature_names)  # 传入特征名
        self.model_params = model_params
        self.training_params = training_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = []
        valid_names = []

        # 验证集
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [val_data]
            valid_names = ['valid']

        callbacks = [
            # 早停机制
            lgb.early_stopping(
                stopping_rounds=self.training_params['early_stopping_rounds']),
            # 每 100 轮输出一次日志
            lgb.log_evaluation(100)
        ]

        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            num_boost_round=self.training_params['num_boost_round'],  # 训练轮数
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def _raw_feature_importance(self, importance_type='gain'):
        if self.model is None:
            return None
        # LightGBM 返回 array，顺序 = feature_name 顺序（因为我们设置了 feature_name!）
        return self.model.feature_importance(importance_type=importance_type)
