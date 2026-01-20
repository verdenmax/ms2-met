from .base_model import BaseModel
from .lgb_model import LGBModel
from .xgb_model import XGBModel
# 未来可添加：from .xgb_model import XGBModel, etc.


class ModelManager:
    @staticmethod
    def create(config: dict) -> BaseModel:
        model_type = config['model'].get('type', 'lightgbm').lower()
        feature_cols = config['data']['feature_cols']

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
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
