from .base_model import BaseModel
from .lgb_model import LGBModel
from .xgb_model import XGBModel
from .sklearn_lr_model import SklearnLRModel
from .sklearn_rf_model import SklearnRFModel
import logging
# 未来可添加：from .xgb_model import XGBModel, etc.


class ModelManager:
    @staticmethod
    def create(config: dict, feature_names: list = None) -> BaseModel:
        """Build the model object specified by `config`.

        Args:
            config: full yaml config dict (model.type / model.params /
                training).
            feature_names: resolved feature column list (after
                resolve_feature_cols expanded the yaml `[]` default into
                the auto-detected list). If omitted, falls back to
                config['data']['feature_cols'] for backward compatibility
                with callers that explicitly populate it.

        Regression for 2026-06-03 runtime bug: when `feature_cols: []`
        is in the yaml (auto-detect mode), reading config['data']
        ['feature_cols'] returns the empty literal, so lgb.Dataset
        receives `feature_name=[]` and crashes with "Length of
        feature_name(0) and num_feature(N) don't match".
        """
        model_type = config['model'].get('type', 'lightgbm').lower()
        if feature_names is None:
            feature_names = config['data']['feature_cols']

        logging.info(f"加载 {model_type} 模型")
        if model_type == 'lightgbm':
            return LGBModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_names,
            )
        elif model_type == 'xgboost':
            return XGBModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_names,
            )
        elif model_type == 'sklearn_rf':
            return SklearnRFModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_names,
            )
        elif model_type == 'sklearn_lr':
            return SklearnLRModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_names,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
