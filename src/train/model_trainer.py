"""Training, tuning and honest evaluation.

Two rules shape this module:

1. Nothing is scored on data that was used to fit it. Hyperparameters are tuned
   by cross-validation, models are compared out-of-fold, blend weights are fitted
   on out-of-fold predictions, and the validation split is touched exactly once,
   at the end, as a final check.
2. The reported metric is the one being optimised. The competition scores RMSLE,
   so the target is log-transformed and the in-CV objective (neg MSE on the
   transformed target) is then numerically identical to MSLE.
"""

import logging
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import exp, log1p
from scipy.optimize import nnls
from scipy.special import boxcox1p, inv_boxcox1p
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from conf.model_config import MODEL_PARAM_GRIDS, grid_size


# -- metrics -------------------------------------------------------------------

def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error -- the competition metric."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 1, None)
    y_true = np.asarray(y_true, dtype=float).ravel()
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def target_transformer(method: str):
    """Transformer applied to the target before fitting, inverted on predict."""
    if method == "log":
        return FunctionTransformer(func=log1p, inverse_func=exp, validate=True)
    if method == "boxcox":
        return FunctionTransformer(
            func=lambda x: boxcox1p(x, 0.15),
            inverse_func=lambda x: inv_boxcox1p(x, 0.15),
            validate=True,
        )
    if method == "yeojohnson":
        return PowerTransformer(method="yeo-johnson")
    if method == "none":
        return FunctionTransformer(validate=True)
    raise ValueError(f"Unknown target_transform '{method}'")


# -- training ------------------------------------------------------------------

class ModelTrainer:
    """Tunes, fits, and compares models on a shared evaluation protocol."""

    def __init__(self, pipeline_manager, cfg, logger: logging.Logger = None) -> None:
        self.pm = pipeline_manager
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.transformer = target_transformer(cfg.training.target_transform)
        self.cv = KFold(
            n_splits=cfg.training.cv_folds,
            shuffle=True,
            random_state=cfg.training.random_state,
        )

    def _wrap(self, regressor) -> TransformedTargetRegressor:
        return TransformedTargetRegressor(regressor=regressor, transformer=self.transformer)

    def tune(self, class_name: str, model, X, y) -> TransformedTargetRegressor:
        """Fit one model, tuning it by cross-validation if enabled."""
        pipeline = self.pm.build_pipeline(model)
        grid = MODEL_PARAM_GRIDS.get(class_name)

        if not self.cfg.training.tune or not grid:
            estimator = self._wrap(pipeline)
            estimator.fit(X, y)
            return estimator

        # RandomizedSearchCV wastes work (and warns) when n_iter exceeds the grid.
        n_iter = min(self.cfg.training.n_iter, grid_size(grid))
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=grid,
            n_iter=n_iter,
            cv=self.cfg.training.cv_folds,
            n_jobs=-1,
            refit=True,
            scoring="neg_mean_squared_error",
        )
        estimator = self._wrap(search)
        estimator.fit(X, y)
        self.logger.info(
            f"{class_name}: tuned over {n_iter} candidates -> {estimator.regressor_.best_params_}"
        )
        return estimator

    def fit_base_models(self, models: Dict[str, object], X, y) -> Dict[str, TransformedTargetRegressor]:
        fitted = {}
        for class_name, model in models.items():
            self.logger.info(f"Training {class_name}...")
            fitted[class_name] = self.tune(class_name, model, X, y)
        return fitted

    def best_estimators(self, fitted: Dict[str, TransformedTargetRegressor]) -> List[Tuple[str, object]]:
        """The tuned estimator out of each fitted model, as (name, estimator) pairs."""
        pairs = []
        for class_name, wrapped in fitted.items():
            regressor = wrapped.regressor_
            pipeline = getattr(regressor, "best_estimator_", regressor)
            pairs.append((class_name.lower(), clone(pipeline.steps[-1][1])))
        return pairs

    def build_stacking(self, base_estimators, X, y) -> TransformedTargetRegressor:
        """Stack with out-of-fold meta-features.

        `cv` is what makes this correct: a meta-learner trained on in-sample base
        predictions learns to trust models that have merely memorised the training
        rows, and reliably underperforms its own base models.
        """
        stack = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=self.cv,
            passthrough=False,
            n_jobs=-1,
        )
        estimator = self._wrap(self.pm.build_pipeline(stack))
        estimator.fit(X, y)
        return estimator

    def build_voting(self, base_estimators, X, y) -> TransformedTargetRegressor:
        estimator = self._wrap(self.pm.build_pipeline(VotingRegressor(estimators=base_estimators)))
        estimator.fit(X, y)
        return estimator

    # -- evaluation ------------------------------------------------------------

    def _unfitted_copy(self, wrapped: TransformedTargetRegressor) -> TransformedTargetRegressor:
        regressor = wrapped.regressor_
        pipeline = getattr(regressor, "best_estimator_", regressor)
        return self._wrap(clone(pipeline))

    def out_of_fold(self, fitted: Dict[str, TransformedTargetRegressor], X, y) -> Dict[str, np.ndarray]:
        """Out-of-fold predictions on the training split, one pass per model.

        Reused for both the model comparison and the blend weights, so the extra
        fits are paid for once.
        """
        oof = {}
        for name, wrapped in fitted.items():
            oof[name] = cross_val_predict(self._unfitted_copy(wrapped), X, y, cv=self.cv, n_jobs=1)
            self.logger.info(f"out-of-fold predictions done: {name}")
        return oof

    def comparison_table(self, oof: Dict[str, np.ndarray], y,
                         val_scores: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """Rank models out-of-fold, with the fold-to-fold spread alongside.

        The spread matters more than the ranking: on ~1,200 rows it routinely
        exceeds the gap between first and last place, and a ranking inside the
        noise is not a result.
        """
        folds = list(self.cv.split(np.zeros(len(y))))
        y = pd.Series(np.asarray(y, dtype=float).ravel())

        rows = []
        for name, preds in oof.items():
            fold_scores = [rmsle(y.iloc[te], preds[te]) for _, te in folds]
            row = {
                "model": name,
                "oof_rmsle": rmsle(y, preds),
                "fold_std": float(np.std(fold_scores)),
            }
            if val_scores and name in val_scores:
                row["val_rmsle"] = val_scores[name]["rmsle"]
                row["val_rmse"] = val_scores[name]["rmse"]
            rows.append(row)

        return pd.DataFrame(rows).sort_values("oof_rmsle").reset_index(drop=True)

    def blend_weights(self, oof: Dict[str, np.ndarray], y) -> Tuple[List[str], np.ndarray]:
        """Non-negative least squares weights, fitted on out-of-fold predictions.

        Fitting weights on validation scores and then reporting the blend's score
        on that same split measures nothing. Non-negativity keeps the result a
        genuine weighted average; the fit is in log space to match the metric.
        """
        names = list(oof)
        A = np.column_stack([np.log1p(np.clip(oof[n], 1, None)) for n in names])
        b = np.log1p(np.asarray(y, dtype=float).ravel())

        w, _ = nnls(A, b)
        if w.sum() == 0:
            w = np.ones(len(names))
        weights = w / w.sum()

        for name, weight in sorted(zip(names, weights), key=lambda t: -t[1]):
            self.logger.info(f"  blend weight {name:28s} {weight:.4f}")
        return names, weights

    @staticmethod
    def blend_predict(fitted: Dict[str, TransformedTargetRegressor], X,
                      names: List[str], weights: np.ndarray) -> np.ndarray:
        """Weighted geometric blend -- arithmetic in log space, matching RMSLE."""
        logp = np.column_stack([
            np.log1p(np.clip(np.asarray(fitted[n].predict(X), dtype=float).ravel(), 1, None))
            for n in names
        ])
        return np.expm1(logp @ weights)
