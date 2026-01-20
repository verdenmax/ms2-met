from .base_model import BaseModel
from .lgb_model import LGBModel
from .xgb_model import XGBModel
from .sklearn_lr_model import SklearnLRModel
from .sklearn_rf_model import SklearnRFModel
import logging
# 未来可添加：from .xgb_model import XGBModel, etc.


class ModelManager:
    @staticmethod
    def create(config: dict) -> BaseModel:
        model_type = config['model'].get('type', 'lightgbm').lower()
        feature_cols = config['data']['feature_cols']

        logging.info(f"加载 {model_type} 模型")
        if model_type == 'lightgbm':
            return LGBModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_cols,
            )
        elif model_type == 'xgboost':
            return XGBModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_cols,
            )
        elif model_type == 'sklearn_rf':
            return SklearnRFModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_cols,
            )
        elif model_type == 'sklearn_lr':
            return SklearnLRModel(
                model_params=config['model']['params'],
                training_params=config['training'],
                feature_names=feature_cols,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
