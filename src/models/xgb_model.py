import xgboost as xgb
import numpy as np
from .base_model import BaseModel


class XGBModel(BaseModel):
    def __init__(self, model_params, training_params, feature_names):
        super().__init__(feature_names)  # 入特征名
        self.model_params = model_params
        self.training_params = training_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=self.feature_names)
        evals = [(dtrain, 'train')]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))

        self.model = xgb.train(
            params=self.model_params,
            dtrain=dtrain,
            num_boost_round=self.training_params['num_boost_round'],
            evals=evals,
            early_stopping_rounds=self.training_params['early_stopping_rounds'],
            verbose_eval=100
        )

    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def _raw_feature_importance(self, importance_type='gain'):
        if self.model is None:
            return None
        # XGBoost 返回 dict {feature_name: imp}
        return self.model.get_score(importance_type=importance_type)
