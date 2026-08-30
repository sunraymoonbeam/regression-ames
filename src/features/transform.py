"""Feature engineering.

Builds the aggregate and age features, and -- critically -- registers them in the
feature-group registry. A column that exists in the frame but is missing from the
registry is silently discarded by the ColumnTransformer downstream, because
`remainder='drop'` is the default. That failure mode is invisible: the model just
quietly trains on fewer columns.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig

from conf.feature_config import AGGREGATE_FEATURES, AGE_FEATURES, BATH_WEIGHTS


class FeatureTransformer:
    """Creates derived features and keeps the feature registry consistent."""

    def __init__(self, df: pd.DataFrame, features: Dict[str, List[str]],
                 cfg: DictConfig, logger: logging.Logger = None) -> None:
        self.df = df.copy()
        self.features = {k: list(v) for k, v in features.items()}
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self._consumed: List[str] = []

    # -- registry helpers -------------------------------------------------------

    def _register(self, group: str, cols: List[str]) -> None:
        self.features[group] += [c for c in cols if c not in self.features[group]]

    def _deregister(self, cols: List[str]) -> None:
        for group in self.features:
            self.features[group] = [c for c in self.features[group] if c not in cols]

    def _sum_present(self, cols: List[str]) -> Tuple[pd.Series, List[str]]:
        """Sum whichever component columns survived earlier cleaning steps."""
        present = [c for c in cols if c in self.df.columns]
        return self.df[present].sum(axis=1), present

    # -- feature construction ---------------------------------------------------

    def build_aggregates(self) -> None:
        created = []
        for name, components in AGGREGATE_FEATURES.items():
            values, used = self._sum_present(components)
            if not used:
                continue
            self.df[name] = values
            self._consumed += used
            created.append(name)

        bath_parts = [c for c in BATH_WEIGHTS if c in self.df.columns]
        if bath_parts:
            self.df["TotalBath"] = sum(self.df[c] * BATH_WEIGHTS[c] for c in bath_parts)
            self._consumed += bath_parts
            created.append("TotalBath")

        self._register("continuous", created)
        self.logger.info(f"Created aggregate features: {created}")

    def build_age_features(self) -> None:
        created = []
        for name, (later, earlier) in AGE_FEATURES.items():
            if later in self.df.columns and earlier in self.df.columns:
                self.df[name] = self.df[later] - self.df[earlier]
                self._consumed.append(earlier)
                created.append(name)

        self._register("continuous", created)
        self.logger.info(f"Created age features: {created}")

    def drop_consumed(self) -> None:
        """Remove the source columns now represented by the derived features."""
        cols = sorted({c for c in self._consumed if c in self.df.columns})
        if not cols:
            return
        self.df = self.df.drop(columns=cols)
        self._deregister(cols)
        self.logger.info(f"Consumed {len(cols)} source columns: {cols}")

    def verify_registry(self) -> None:
        """Fail loudly if the registry and the frame have drifted apart.

        This is the guard for the silent-drop failure mode described in the module
        docstring -- an assertion here is much cheaper than wondering why a model
        underperforms.
        """
        covered = set().union(*self.features.values())
        uncovered = set(self.df.columns) - covered
        stale = covered - set(self.df.columns)
        if uncovered:
            raise AssertionError(
                f"Columns present in the data but missing from the feature registry "
                f"(they would be silently dropped before every model): {sorted(uncovered)}"
            )
        if stale:
            raise AssertionError(
                f"Feature registry references columns that no longer exist: {sorted(stale)}"
            )

    def feature_engineer(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if self.cfg.feature_engineering.aggregates:
            self.build_aggregates()
        if self.cfg.feature_engineering.age_features:
            self.build_age_features()
        self.drop_consumed()
        self.verify_registry()

        self.logger.info(
            f"Feature engineering complete: {self.df.shape[1]} columns "
            f"({len(self.features['continuous'])} continuous, "
            f"{len(self.features['categorical'])} categorical, "
            f"{len(self.features['ordinal'])} ordinal)"
        )
        return self.df, self.features
