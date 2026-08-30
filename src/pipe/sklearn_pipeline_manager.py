"""Construction of the preprocessing pipeline.

Every learned transformation lives here rather than being applied to the frame
up front, so each one is refit on every CV fold. Fitting an imputer or a scaler
once on the whole dataset and then cross-validating over it leaks information
between folds and produces an optimistic score.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


class NeighborhoodMedianImputer(BaseEstimator, TransformerMixin):
    """Impute LotFrontage with the median of its Neighborhood.

    Lot frontage is largely set by the street layout of a neighbourhood, so the
    neighbourhood median is a much better estimate than a global one -- good
    enough that it is worth keeping a column that is 16.6% missing rather than
    discarding it.

    The medians are *learned*, so this has to be a transformer fit inside the
    pipeline rather than a one-off pass over the whole frame. Unseen groups and
    groups that are entirely missing fall back to the global median.
    """

    def __init__(self, target_col: str = "LotFrontage", group_col: str = "Neighborhood") -> None:
        self.target_col = target_col
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y=None) -> "NeighborhoodMedianImputer":
        self.active_ = (
            self.group_col is not None
            and self.target_col in X.columns
            and self.group_col in X.columns
        )
        if self.active_:
            self.medians_ = X.groupby(self.group_col)[self.target_col].median()
            self.global_median_ = X[self.target_col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.active_:
            return X
        X = X.copy()
        fill = X[self.group_col].map(self.medians_).fillna(self.global_median_)
        X[self.target_col] = X[self.target_col].fillna(fill)
        return X


def _numeric_scaler(method: str):
    scalers = {
        # Corrects predictor skew and standardises in one step. Unlike log1p it
        # tolerates the zeros in TotalPorch and the negatives RemodAge can take.
        "yeo-johnson": PowerTransformer(method="yeo-johnson", standardize=True),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if method not in scalers:
        raise ValueError(f"Unknown numeric_scaling '{method}'. Options: {sorted(scalers)}")
    return scalers[method]


class TransformerPipelineManager:
    """Builds the ColumnTransformer and wraps models in a full pipeline."""

    def __init__(self, features: Dict[str, List[str]], cfg, logger: logging.Logger = None) -> None:
        self.features = features
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = self.create_preprocessor()

    def verify_coverage(self, columns) -> None:
        """ColumnTransformer drops anything not named in a transformer."""
        covered = set().union(*self.features.values())
        uncovered = set(columns) - covered
        if uncovered:
            raise AssertionError(
                f"Columns not covered by the preprocessor (they would be silently "
                f"dropped before every model): {sorted(uncovered)}"
            )

    def create_preprocessor(self) -> ColumnTransformer:
        pre = self.cfg.preprocessing

        numeric_transformer = make_pipeline(
            SimpleImputer(strategy=pre.numeric_imputation),
            _numeric_scaler(pre.numeric_scaling),
        )
        categorical_transformer = make_pipeline(
            SimpleImputer(strategy=pre.categorical_imputation),
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        )
        ordinal_transformer = make_pipeline(
            SimpleImputer(strategy=pre.ordinal_imputation),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        )

        self.logger.info(
            f"Preprocessor: {len(self.features['continuous'])} continuous "
            f"({pre.numeric_imputation} impute, {pre.numeric_scaling}), "
            f"{len(self.features['categorical'])} categorical (onehot), "
            f"{len(self.features['ordinal'])} ordinal"
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.features["continuous"]),
                ("cat", categorical_transformer, self.features["categorical"]),
                ("ord", ordinal_transformer, self.features["ordinal"]),
            ]
        )

    def build_pipeline(self, model) -> Pipeline:
        """Front-end plus estimator.

        A factory rather than an inline `make_pipeline` at each call site, so the
        LotFrontage imputer cannot be accidentally left out of one model's
        pipeline. The estimator keeps its usual step name, so parameter grids
        prefixed with the lowercased class name still apply.

        No `memory=` cache: joblib.Memory hashes each step to build a cache key
        and cannot hash a locally-defined transformer class, which makes every
        fit in a search fail. The fits are cheap enough not to need it.
        """
        return make_pipeline(
            NeighborhoodMedianImputer(group_col=self.cfg.preprocessing.lot_frontage_group_column),
            self.preprocessor,
            model,
        )
