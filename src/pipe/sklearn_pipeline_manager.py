from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from category_encoders import TargetEncoder
from sklearn import set_config
import itertools
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

set_config(
    display="diagram", transform_output="pandas"
)  # Transformers returns pandas dataframes rather then numpy arrays


class TransformerPipelineManager:
    """
    This class is responsible for creating, fitting and evaluating a machine learning pipeline.

    Attributes
    ----------
    logger : Logger
        a logger for logging events during model training

    Methods
    -------
    create_numeric_transformer(scaling_method, numerical_imputation_method=None):
        Create a transformer for numerical features.
    create_categorical_transformer(categorical_encoding_method, categorical_imputation_method=None):
        Create a transformer for categorical features.
    create_ordinal_transformer():
        Create a transformer for ordinal features.
    create_transformations(scaling_options, encoding_options, numeric_columns, categorical_columns, ordinal_columns):
        Create all combinations of transformations for the given options and columns.
    create_pipeline():
        Initialize a Pipeline with placeholders for the data transformations and classifier.
    create_param_grid(transformations, models, MODEL_PARAMETERS):
        Define the parameter grid for RandomizedSearchCV.
    fit_pipeline(X_train, y_train):
        Fit the pipeline using RandomizedSearchCV.
    evaluate_pipeline(X_test, y_test):
        Evaluate the best model pipeline.
    """

    def __init__(self, logger):
        """
        Initialize the PipelineTransformer with a logger.
        """
        self.logger = logger

    def create_numeric_transformer(
        self, scaling_method, numerical_imputation_method=None
    ):
        steps = (
            [
                (
                    "imputer",
                    SimpleImputer(strategy=numerical_imputation_method or "median"),
                )
            ]
            if numerical_imputation_method
            else []
        )
        steps.append(
            (
                "scaler",
                MinMaxScaler() if scaling_method == "minmax" else StandardScaler(),
            )
        )
        return Pipeline(steps=steps)

    def create_categorical_transformer(
        self, categorical_encoding_method, categorical_imputation_method=None
    ):
        steps = (
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=categorical_imputation_method or "constant",
                        fill_value="missing",
                    ),
                )
            ]
            if categorical_imputation_method
            else []
        )
        steps.append(
            (
                "encoder",
                (
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    if categorical_encoding_method == "onehot"
                    else TargetEncoder()
                ),
            )
        )
        return Pipeline(steps=steps)

    def create_ordinal_transformer(self):
        return Pipeline(steps=[("ordinal", OrdinalEncoder())])

    def create_transformations(
        self,
        scaling_options,
        encoding_options,
        numeric_columns,
        categorical_columns,
        ordinal_columns,
    ):
        numeric_transformers = [
            self.create_numeric_transformer(option) for option in scaling_options
        ]
        categorical_transformers = [
            self.create_categorical_transformer(option) for option in encoding_options
        ]
        ordinal_transformer = self.create_ordinal_transformer()

        transformations = []
        for num_transformer, cat_transformer in itertools.product(
            numeric_transformers, categorical_transformers
        ):
            transformation = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, numeric_columns),
                    ("cat", cat_transformer, categorical_columns),
                    ("ord", ordinal_transformer, ordinal_columns),
                ]
            )
            transformations.append(transformation)

        return transformations

    def create_pipeline(self):
        self.logger.info("Creating pipeline...")
        self.pipe = Pipeline(
            steps=[
                (
                    "data_transformations",
                    "passthrough",
                ),  # 'passthrough' means that this step will be replaced
                (
                    "clf",
                    "passthrough",
                ),  # 'passthrough' means that this step will be replaced
            ]
        )

    def create_param_grid(self, transformations, models, MODEL_PARAMETERS):
        self.params_grid = [
            {
                "data_transformations": transformations,
                "clf": [model],
                **{
                    f"clf__{param}": values
                    for param, values in MODEL_PARAMETERS[model_name].items()
                },
            }
            for model_name, model in models.items()
            if model_name in MODEL_PARAMETERS
        ]

    def fit_pipeline(self, X_train, y_train, n_iter, tune_metric="recall"):
        self.logger.info("Fitting pipeline...")
        self.best_model_pipeline = RandomizedSearchCV(
            estimator=self.pipe,
            param_distributions=self.params_grid,
            n_iter=n_iter,
            scoring=tune_metric,
            n_jobs=-1,
            cv=5,
            random_state=21,
            error_score="raise",
            return_train_score=False,
        )
        self.best_model_pipeline.fit(X_train, y_train)

        self.logger.info(
            f"Best Data Pipeline: {self.best_model_pipeline.best_estimator_[0]}"
        )
        self.logger.info(
            f"Best Classifier: {self.best_model_pipeline.best_estimator_[1]}"
        )

    def evaluate_pipeline(self, X_test, y_test):
        if self.best_model_pipeline is None:
            raise Exception(
                "The pipeline has not been fitted yet. Please call fit_pipeline first."
            )

        y_pred = self.best_model_pipeline.predict(X_test)
        result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }
        self.logger.info(f"Performance: {result}")
        return result
