"""Regressor registry, default parameters and search grids.

Grid keys use the estimator's step name as produced by `make_pipeline`, which is
the lowercased class name -- so they can be handed straight to a search over the
full pipeline.
"""

import numpy as np

# Short name used in config.yaml -> scikit-learn style class name.
MODEL_NAME_MAPPING = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "bayesian_ridge": "BayesianRidge",
    "gbr": "GradientBoostingRegressor",
    "catboost": "CatBoostRegressor",
    "lgbm": "LGBMRegressor",
}

# Pipeline step name for each model, used to prefix its grid.
MODEL_STEP_NAMES = {
    "Ridge": "ridge",
    "Lasso": "lasso",
    "ElasticNet": "elasticnet",
    "BayesianRidge": "bayesianridge",
    "GradientBoostingRegressor": "gradientboostingregressor",
    "CatBoostRegressor": "catboostregressor",
    "LGBMRegressor": "lgbmregressor",
}

DEFAULT_MODEL_PARAMETERS = {
    "Ridge": {"alpha": 1.0},
    "Lasso": {"alpha": 0.0005, "max_iter": 20000, "random_state": 42},
    "ElasticNet": {"alpha": 0.0005, "l1_ratio": 0.9, "max_iter": 20000, "random_state": 42},
    "BayesianRidge": {"max_iter": 300},
    "GradientBoostingRegressor": {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "max_features": "sqrt",
        "min_samples_leaf": 15,
        "min_samples_split": 10,
        "loss": "huber",       # robust to the price outliers that remain
        "random_state": 5,
    },
    "CatBoostRegressor": {"n_estimators": 1000, "depth": 3, "logging_level": "Silent"},
    "LGBMRegressor": {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "verbose": -1,
    },
}

MODEL_PARAM_GRIDS = {
    "Ridge": {
        "ridge__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        "ridge__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    },
    "Lasso": {
        "lasso__alpha": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1],
        "lasso__max_iter": [5000, 20000],
        "lasso__selection": ["cyclic", "random"],
    },
    "ElasticNet": {
        "elasticnet__alpha": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1],
        "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        "elasticnet__max_iter": [5000, 20000],
    },
    "BayesianRidge": {
        "bayesianridge__alpha_1": [1e-7, 1e-6, 1e-5, 1e-4],
        "bayesianridge__alpha_2": [1e-7, 1e-6, 1e-5, 1e-4],
        "bayesianridge__lambda_1": [1e-7, 1e-6, 1e-5, 1e-4],
        "bayesianridge__lambda_2": [1e-7, 1e-6, 1e-5, 1e-4],
        "bayesianridge__max_iter": [100, 200, 300, 400, 500],
    },
    "GradientBoostingRegressor": {
        "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.3],
        "gradientboostingregressor__n_estimators": [100, 500, 1000, 2000, 3000],
        "gradientboostingregressor__max_depth": [3, 4, 6, 9],
        "gradientboostingregressor__min_samples_split": [2, 5, 10],
        "gradientboostingregressor__min_samples_leaf": [1, 2, 5, 15],
    },
    "CatBoostRegressor": {
        "catboostregressor__n_estimators": [100, 300, 500, 1000, 1300, 1600],
        "catboostregressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "catboostregressor__l2_leaf_reg": [0.001, 0.01, 0.1],
        "catboostregressor__random_strength": [0.25, 0.5, 1],
        "catboostregressor__depth": [3, 6, 9],  # CatBoost's own name; "max_depth" is a synonym and setting both errors
        "catboostregressor__min_child_samples": [2, 5, 10, 15, 20],
        "catboostregressor__border_count": [32, 64, 128, 255],
    },
    "LGBMRegressor": {
        "lgbmregressor__max_depth": [3, 5, 8, 10],
        "lgbmregressor__learning_rate": [0.001, 0.01, 0.1, 0.2],
        "lgbmregressor__n_estimators": [100, 300, 500, 1000, 1500],
        "lgbmregressor__reg_alpha": [0.0001, 0.001, 0.01],
        "lgbmregressor__reg_lambda": [0, 0.0001, 0.001, 0.01],
        "lgbmregressor__colsample_bytree": [0.4, 0.6, 0.8],
        "lgbmregressor__min_child_samples": [5, 10, 20, 25],
        "lgbmregressor__num_leaves": [31, 62, 124, 248],
    },
}


def grid_size(param_grid: dict) -> int:
    """Number of distinct combinations in a discrete grid.

    RandomizedSearchCV warns (and wastes work) when n_iter exceeds this.
    """
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return int(total)
