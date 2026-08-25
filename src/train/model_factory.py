"""
This script contains a class that creates machine learning models. 

List of models available for training:
- `GBC` (Gradient Boosting Classifier): A model that uses the Gradient Boosting framework.
- `Cat` (CatBoost Classifier): A model that uses gradient boosting on decision trees, especially powerful in handling categorical features.
- `XGB` (XGBoost Classifier): An implementation of gradient boosting machines designed to be highly efficient, flexible and portable.
- `RFC` (Random Forest Classifier): An ensemble learning method where a few weak models combine to form a powerful model.
- `LR` (Logistic Regression): A simple machine learning classification algorithm used to predict the probability of a categorical dependent variable.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# from catboost import CatBoostClassifier # Catboost has issues with the current environment

from xgboost import XGBClassifier
from typing import Dict, Optional, List, Any


class ModelFactory:
    """
    A class used to create machine learning models.

    ## Attributes
    --------------------------------
    * models : dict
        a dictionary of model names and their corresponding classes

    ## Methods
    --------------------------------
    1. __init__():
        Initializes the ModelFactory class with a dictionary of model names and their corresponding classes.

    2. get_models(model_names=None):
        Returns a dictionary of model names and their instances. If no model names are specified, all models are returned.

    3. list_available_models():
        Returns a list of all model names.
    """

    def __init__(self):
        """
        Initializes the ModelFactory class with a dictionary of model names and their corresponding classes.
        """
        self.models = {
            "GradientBoostingClassifier": GradientBoostingClassifier,
            #  "CatBoostClassifier": CatBoostClassifier,
            "XGBClassifier": XGBClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
        }

    def get_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieves instances of the specified models. If no models are specified, or 'all' is specified, instances of all available models are returned.

        Args:
            model_names (Optional[List[str]]): A list of model names to retrieve. If None or contains 'all', all models are returned.

        Returns:
            Dict[str, Any]: A dictionary of model names and their instantiated objects.
        """
        if model_names is None or "all" in model_names:
            return {name: model() for name, model in self.models.items()}
        else:
            return {
                name: self.models[name]() for name in model_names if name in self.models
            }

    def list_available_models(self) -> List[str]:
        """
        Lists the names of all available models in the factory.

        Returns:
            List[str]: A list of available model names.
        """
        return list(self.models.keys())
