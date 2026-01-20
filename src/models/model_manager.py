from .base_model import BaseModel
from .lgb_model import LGBModel
# 未来可添加：from .xgb_model import XGBModel, etc.


class ModelManager:
    @staticmethod
    def create(config: dict) -> BaseModel:
        model_type = config['model'].get('type', 'lightgbm').lower()

        if model_type == 'lightgbm':
            return LGBModel(
                model_params=config['model']['params'],
                training_params=config['training']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
