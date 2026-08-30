"""Regressor construction."""

import logging
from typing import Dict, List

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge

from conf.model_config import DEFAULT_MODEL_PARAMETERS, MODEL_NAME_MAPPING

MODEL_REGISTRY = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "BayesianRidge": BayesianRidge,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    "LGBMRegressor": LGBMRegressor,
}


class ModelFactory:
    """Builds regressors by name, seeded with their default parameters."""

    def __init__(self, logger: logging.Logger = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def get_models(self, short_names: List[str]) -> Dict[str, object]:
        """Map config short names ('gbr') to instantiated regressors.

        Returns an ordered dict keyed by class name, so downstream code can look
        up parameter grids without carrying the short names around.
        """
        if "all" in short_names:
            short_names = list(MODEL_NAME_MAPPING)

        models = {}
        for short in short_names:
            class_name = MODEL_NAME_MAPPING.get(short)
            if class_name is None:
                self.logger.warning(
                    f"Unknown model '{short}', skipping. Available: {sorted(MODEL_NAME_MAPPING)}"
                )
                continue
            params = dict(DEFAULT_MODEL_PARAMETERS.get(class_name, {}))
            models[class_name] = MODEL_REGISTRY[class_name](**params)

        if not models:
            raise ValueError(f"No valid models selected from {short_names}")
        self.logger.info(f"Instantiated {len(models)} models: {list(models)}")
        return models

    @staticmethod
    def list_available_models() -> List[str]:
        return sorted(MODEL_NAME_MAPPING)
