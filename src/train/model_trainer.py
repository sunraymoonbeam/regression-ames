import warnings
import pickle
from typing import List, Tuple, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from logging import Logger

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    A class used to train machine learning models.

    Attributes:
    -----------
    model : BaseEstimator
        The machine learning model to be trained.
    parameters : Dict[str, Any]
        The parameters of the model.
    logger : Logger
        A logger for logging events during model training.

    Methods:
    --------
    set_params(params: Dict[str, Any]) -> None:
        Sets the parameters of the model.

    train(X_train: Any, y_train: Any) -> None:
        Trains the model on the training data.

    train_voting_classifier(models: List[Tuple[str, BaseEstimator]], X_train: Any, y_train: Any) -> VotingClassifier:
        Train a VotingClassifier using the provided models.

    train_stacking_classifier(models: List[Tuple[str, BaseEstimator]], final_estimator: BaseEstimator, X_train: Any, y_train: Any) -> StackingClassifier:
        Train a StackingClassifier using the provided models and final estimator.

    predict(X_test: Any) -> Any:
        Makes predictions on the test data.

    evaluate(y_test: Any, y_pred: Any) -> Dict[str, float]:
        Evaluates the performance of the model on the test data.

    grid_tune_parameters(X_train: Any, y_train: Any, param_grid: Dict[str, List[Any]], tune_metric: str, verbose: int = 0) -> Dict[str, Any]:
        Tunes the hyperparameters of the model to optimize a specified metric using GridSearchCV.

    random_tune_parameters(X_train: Any, y_train: Any, param_grid: Dict[str, List[Any]], tune_metric: str, verbose: int = 0, n_iter: int = 100) -> Dict[str, Any]:
        Tunes the hyperparameters of the model to optimize a specified metric using RandomizedSearchCV.

    save_model(filepath: str = 'models/model.pkl') -> None:
        Saves the trained model to a pickle file.
    """

    def __init__(
        self, model: BaseEstimator, parameters: Dict[str, Any], logger: Logger
    ) -> None:
        """
        Initializes the ModelTrainer class with a model, its parameters, and a logger.

        Parameters:
        -----------
        model : BaseEstimator
            The machine learning model to be trained.
        parameters : Dict[str, Any]
            The parameters of the model.
        logger : Logger
            A logger for logging events during model training.
        """
        self.model = model
        self.parameters = parameters
        self.logger = logger

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters of the model.

        Parameters:
        -----------
        params : Dict[str, Any]
            A dictionary of parameter names and their corresponding values.
        """
        self.parameters = params
        self.model.set_params(**params)

    def train(self, X_train: Any, y_train: Any) -> None:
        """
        Trains the model on the training data.

        Parameters:
        -----------
        X_train : Any
            The training input samples.
        y_train : Any
            The target values (class labels).
        """
        self.logger.info(f"Starting training for {self.model.__class__.__name__}...")
        self.model.set_params(**self.parameters)
        self.model.fit(X_train, y_train)
        self.logger.info(f"Finished training for {self.model.__class__.__name__}.")

    def train_voting_classifier(
        self, models: List[Tuple[str, BaseEstimator]], X_train: Any, y_train: Any
    ) -> VotingClassifier:
        """
        Train a VotingClassifier using the provided models.

        Parameters:
        -----------
        models : List[Tuple[str, BaseEstimator]]
            A list of (str, estimator) tuples, where the string is the name of the estimator.
        X_train : Any
            The training input samples.
        y_train : Any
            The target values (class labels).

        Returns:
        --------
        VotingClassifier
            The trained VotingClassifier.
        """
        voting_clf = VotingClassifier(estimators=models, voting="soft")
        self.logger.info(f"Starting training for VotingClassifier...")
        voting_clf.fit(X_train, y_train)
        self.logger.info(f"Finished training for VotingClassifier.")
        return voting_clf

    def train_stacking_classifier(
        self,
        models: List[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        X_train: Any,
        y_train: Any,
    ) -> StackingClassifier:
        """
        Train a StackingClassifier using the provided models and final estimator.

        Parameters:
        -----------
        models : List[Tuple[str, BaseEstimator]]
            A list of (str, estimator) tuples, where the string is the name of the estimator.
        final_estimator : BaseEstimator
            The final estimator to use. This estimator will be trained on the predictions of the base models.
        X_train : Any
            The training input samples.
        y_train : Any
            The target values (class labels).

        Returns:
        --------
        StackingClassifier
            The trained StackingClassifier.
        """
        stacking_clf = StackingClassifier(
            estimators=models, final_estimator=final_estimator
        )
        self.logger.info(f"Starting training for StackingClassifier...")
        stacking_clf.fit(X_train, y_train)
        self.logger.info(f"Finished training for StackingClassifier.")
        self.model = stacking_clf
        self.parameters = stacking_clf.get_params()
        return stacking_clf

    def predict(self, X_test: Any) -> Any:
        """
        Makes predictions on the test data.

        Parameters:
        -----------
        X_test : Any
            The test input samples.

        Returns:
        --------
        Any
            The predictions made by the model.
        """
        return self.model.predict(X_test)

    def evaluate(self, y_test: Any, y_pred: Any) -> Dict[str, float]:
        """
        Evaluates the performance of the model on the test data.

        Parameters:
        -----------
        y_test : Any
            The true target values.
        y_pred : Any
            The predicted target values by the model.

        Returns:
        --------
        Dict[str, float]
            A dictionary containing the evaluation metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }
        self.logger.info(
            f"{self.model.__class__.__name__} performance - "
            + ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        )
        return metrics

    def grid_tune_parameters(
        self,
        X_train: Any,
        y_train: Any,
        param_grid: Dict[str, List[Any]],
        tune_metric: str,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Tunes the hyperparameters of the model to optimize a specified metric using GridSearchCV.

        Parameters:
        -----------
        X_train : Any
            The training input samples.
        y_train : Any
            The target values (class labels).
        param_grid : Dict[str, List[Any]]
            The parameter grid to explore.
        tune_metric : str
            The metric to optimize.
        verbose : int, optional
            The verbosity level.

        Returns:
        --------
        Dict[str, Any]
            The best parameters found.
        """
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring=tune_metric, verbose=verbose
        )
        grid_search.fit(X_train, y_train)
        self.logger.info(
            f"Tuned parameters for {self.model.__class__.__name__}: {grid_search.best_params_}"
        )
        self.logger.info(
            f"Best cross-validation score for {self.model.__class__.__name__}: {grid_search.best_score_}"
        )
        return grid_search.best_params_

    def random_tune_parameters(
        self,
        X_train: Any,
        y_train: Any,
        param_grid: Dict[str, List[Any]],
        tune_metric: str,
        verbose: int = 0,
        n_iter: int = 100,
    ) -> Dict[str, Any]:
        """
        Tunes the hyperparameters of the model to optimize a specified metric using RandomizedSearchCV.

        Parameters:
        -----------
        X_train : Any
            The training input samples.
        y_train : Any
            The target values (class labels).
        param_grid : Dict[str, List[Any]]
            The parameter grid to explore.
        tune_metric : str
            The metric to optimize.
        verbose : int, optional
            The verbosity level.
        n_iter : int, optional
            The number of parameter settings that are sampled.

        Returns:
        --------
        Dict[str, Any]
            The best parameters found.
        """
        random_search = RandomizedSearchCV(
            self.model,
            param_grid,
            n_iter=n_iter,
            cv=5,
            scoring=tune_metric,
            verbose=verbose,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        self.logger.info(
            f"Tuned parameters for {self.model.__class__.__name__}: {random_search.best_params_}"
        )
        self.logger.info(
            f"Best cross-validation score for {self.model.__class__.__name__}: {random_search.best_score_}"
        )
        return random_search.best_params_

    def save_model(self, filepath: str = "models/model.pkl") -> None:
        """
        Saves the trained model to a pickle file.

        Parameters:
        -----------
        filepath : str, optional
            The path to save the pickle file.

        Returns:
        --------
        None
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"{self.model.__class__.__name__} saved to {filepath}.")
