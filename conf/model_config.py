import numpy as np

"""
This configuration file contains constants for machine learning models.

The constants include:
- Parameters for Gradient Boosting Classifier (GBC), CatBoost Classifier (Cat), 
  XGBoost (XGB), Random Forest Classifier (RFC), and Logistic Regression (LR).
- Parameter grids for hyperparameter tuning of the above models.

Each model has its own dictionary of parameters. For each parameter, a brief description 
is provided in the form of a comment.
"""

# Mapping between model class names and keys in MODEL_PARAM_GRIDS
MODEL_NAME_MAPPING = {
    "GBC": "GradientBoostingClassifier",
    "Cat": "CatBoostClassifier",
    "XGB": "XGBClassifier",
    "RFC": "RandomForestClassifier",
    "LR": "LogisticRegression",
}

# Model parameters
DEFAULT_MODEL_PARAMETERS = {
    "GradientBoostingClassifier": {
        "n_estimators": 100,  # The number of boosting stages to perform
        "learning_rate": 0.1,  # Learning rate shrinks the contribution of each tree
        "max_depth": 3,  # Maximum depth of the individual regression estimators
        "subsample": 0.8,  # The fraction of samples to be used for fitting the individual base learners
        "min_samples_split": 2,  # The minimum number of samples required to split an internal node
        "min_samples_leaf": 1,  # The minimum number of samples required to be at a leaf node
    },
    "CatBoostClassifier": {
        "iterations": 100,  # The maximum number of trees that can be built
        "learning_rate": 0.1,  # The learning rate
        "depth": 3,  # Depth of the tree
    },
    "XGBClassifier": {
        "n_estimators": 100,  # Number of gradient boosted trees
        "learning_rate": 0.1,  # Boosting learning rate
        "max_depth": 3,  # Maximum tree depth
        "gamma": 0,  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        "subsample": 0.8,  # Subsample ratio of the training instances
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    },
    "RandomForestClassifier": {
        "n_estimators": 100,  # The number of trees in the forest
        "max_depth": 3,  # The maximum depth of the tree
        "min_samples_split": 2,  # The minimum number of samples required to split an internal node
        "min_samples_leaf": 1,  # The minimum number of samples required to be at a leaf node
        "max_features": "sqrt",  # The number of features to consider when looking for the best split
    },
    "LogisticRegression": {
        "C": 1.0,  # Inverse of regularization strength
        "penalty": "l2",  # Used to specify the norm used in the penalization
        "fit_intercept": True,  # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function
        "solver": "liblinear",  # Algorithm to use in the optimization problem
    },
}


# Parameter grids for hyperparameter tuning
MODEL_PARAM_GRIDS = {
    "GradientBoostingClassifier": {
        "n_estimators": np.linspace(50, 200, 5, dtype=int),
        "max_depth": np.linspace(3, 10, 5, dtype=int),
        "learning_rate": np.linspace(0.001, 0.1, 10),
        "subsample": np.linspace(0.05, 1, 10),
        "min_samples_split": np.linspace(2, 100, 5, dtype=int),
        "min_samples_leaf": np.linspace(1, 50, 5, dtype=int),
    },
    "CatBoostClassifier": {
        "iterations": np.linspace(50, 200, 5, dtype=int),
        "depth": np.linspace(3, 10, 5, dtype=int),
        "learning_rate": np.linspace(0.001, 0.1, 10),
        "subsample": np.linspace(0.05, 1, 10),
        "min_data_in_leaf": np.linspace(1, 50, 5, dtype=int),
    },
    "XGBClassifier": {
        "n_estimators": np.linspace(50, 200, 5, dtype=int),
        "max_depth": np.linspace(3, 10, 5, dtype=int),
        "learning_rate": np.linspace(0.001, 0.1, 10),
        "gamma": np.linspace(0.1, 1, 5),
        "subsample": np.linspace(0.05, 1, 10),
        "colsample_bytree": np.linspace(0.5, 1, 5),
    },
    "RandomForestClassifier": {
        "n_estimators": np.linspace(50, 200, 5, dtype=int),
        "max_depth": np.linspace(3, 10, 5, dtype=int),
        "min_samples_split": np.linspace(2, 100, 5, dtype=int),
        "min_samples_leaf": np.linspace(1, 50, 5, dtype=int),
        "max_features": ["sqrt", "log2"],
    },
    "LogisticRegression": {
        "C": np.logspace(-2, 1, 5),
        "fit_intercept": [True, False],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "penalty": ["l2", None],
        "l1_ratio": np.linspace(0, 1, 5),
    },
}
