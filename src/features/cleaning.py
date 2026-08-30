"""Cleaning and column-level typing for the Ames dataset.

Everything in here is either a deterministic mapping (recoding, ordinal
encoding, absent -> "None") or a structural column decision made on the shape of
the data rather than on the target. None of it estimates a statistic that is
later applied to held-out rows, so none of it can leak. Anything that *is*
learned -- imputation, scaling, encoding -- lives in the pipeline instead, where
it is refit on every CV fold. See ``src.pipe.sklearn_pipeline_manager``.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conf.feature_config import ORDINAL_MAPPINGS, ORDINAL_FEATURES, RECODE_MAPS


class DataPreprocessor:
    """Applies the cleaning steps and works out which group each column belongs to.

    Attributes:
        df: the combined train+test predictor frame being cleaned.
        features: mapping of 'continuous' / 'categorical' / 'ordinal' to column
            names. This registry is what the ColumnTransformer is built from, so
            it must stay exactly in step with ``df.columns``.
    """

    def __init__(self, x_all: pd.DataFrame, cfg: DictConfig, logger: logging.Logger = None) -> None:
        self.df = x_all.copy()
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.features: Dict[str, List[str]] = {}

    # -- deterministic recoding -------------------------------------------------

    def recode_categoricals(self) -> None:
        """Numeric codes that are really nominal categories."""
        maps = {c: RECODE_MAPS[c] for c in self.cfg.cleaning.recode_as_categorical if c in RECODE_MAPS}
        self.df = self.df.replace(maps)
        self.logger.info(f"Recoded as nominal categories: {list(maps)}")

    def encode_ordinals(self) -> None:
        """Ordered ratings -> integers, worst to best."""
        present = {c: m for c, m in ORDINAL_MAPPINGS.items() if c in self.df.columns}
        self.df = self.df.replace(present)
        self.logger.info(f"Ordinal-encoded {len(present)} rating columns")

    def fill_absent_categories(self) -> None:
        """Where NA means 'the feature is absent', say so explicitly."""
        cols = [c for c in self.cfg.cleaning.absent_means_none if c in self.df.columns]
        if cols:
            self.df[cols] = self.df[cols].fillna("None")
        self.logger.info(f"Filled with the 'None' category (absence, not missingness): {cols}")

    # -- structural column decisions -------------------------------------------

    def build_feature_groups(self) -> None:
        """Split columns into continuous / categorical / ordinal."""
        numeric = self.df.select_dtypes(include=[np.number]).columns
        objects = self.df.select_dtypes(include=["object"]).columns
        ordinal = [c for c in ORDINAL_FEATURES if c in self.df.columns]

        self.features = {
            "continuous": [c for c in numeric if c not in ordinal],
            "categorical": [c for c in objects if c not in ordinal],
            "ordinal": ordinal,
        }

    def drop_columns(self, cols: List[str], reason: str) -> None:
        """Drop columns and keep the feature registry in step."""
        cols = [c for c in cols if c in self.df.columns]
        if not cols:
            return
        self.df = self.df.drop(columns=cols)
        for group in self.features:
            self.features[group] = [c for c in self.features[group] if c not in cols]
        self.logger.info(f"Dropped {len(cols)} columns ({reason}): {cols}")

    def drop_collinear(self) -> None:
        """Drop true near-duplicates only.

        Every model downstream is either L2/L1-regularised or a tree ensemble, so
        collinearity is handled by the penalty or simply irrelevant. Pruning one
        of each correlated pair would only discard signal.
        """
        self.drop_columns(list(self.cfg.cleaning.collinear_to_drop), "near-duplicate")

    def drop_near_constant(self) -> None:
        """Drop columns where a single value dominates."""
        threshold = self.cfg.cleaning.near_constant_threshold
        near_constant = [
            c for c in self.df.columns
            if self.df[c].value_counts(normalize=True, dropna=False).iloc[0] > threshold
        ]
        self.drop_columns(near_constant, f">{threshold:.0%} one value")

    def drop_configured(self) -> None:
        self.drop_columns(list(self.cfg.cleaning.drop_columns), "configured drop")

    # -- orchestration ----------------------------------------------------------

    def preprocess(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Run the full cleaning sequence.

        Returns the cleaned frame and the feature-group registry.
        """
        self.recode_categoricals()
        self.encode_ordinals()
        self.fill_absent_categories()
        self.build_feature_groups()

        self.drop_collinear()
        self.drop_near_constant()
        self.drop_configured()

        remaining = int((self.df.isnull().sum() > 0).sum())
        self.logger.info(
            f"Cleaning complete: {self.df.shape[1]} columns, {remaining} still carrying "
            f"missing values (imputed inside the pipeline, per fold)"
        )
        return self.df, self.features


def outlier_index(raw_train: pd.DataFrame, cfg: DictConfig, logger: logging.Logger = None) -> pd.Index:
    """Row labels of training outliers, by threshold on the ORIGINAL columns.

    Returns an index rather than a filtered frame on purpose. These rows must be
    dropped from the training split *after* the train/validation split -- dropping
    them beforehand also cleans the validation set, which flatters the score for a
    thing you cannot do at inference time.
    """
    logger = logger or logging.getLogger(__name__)
    if not cfg.outliers.enabled:
        return pd.Index([])

    mask = pd.Series(False, index=raw_train.index)
    for column, threshold in cfg.outliers.thresholds.items():
        if column in raw_train.columns:
            mask |= raw_train[column] > threshold

    idx = raw_train.index[mask]
    logger.info(f"{len(idx)} outlier rows flagged (removed from the training split only)")
    return idx
