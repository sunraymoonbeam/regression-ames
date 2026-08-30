"""Loading the Ames competition CSVs."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

KAGGLE_URL = (
    "https://www.kaggle.com/competitions/"
    "house-prices-advanced-regression-techniques/data"
)


class DataLoader:
    """Reads train.csv and test.csv and puts both on a shared integer index.

    The competition ships the predictors and the target in separate files. Both
    are indexed by ``Id - 1`` so a row keeps the same label whether it is being
    looked at as training data, validation data or a test-set prediction.
    """

    def __init__(self, cfg, logger: logging.Logger = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.raw_dir = Path(cfg.data.raw_dir)

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the competition data from {KAGGLE_URL} "
                f"and place train.csv / test.csv in {self.raw_dir}/ at the project root."
            )
        return pd.read_csv(path)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load the raw data.

        Returns:
            raw_train: the training frame, target included, original columns intact.
                Kept so outlier thresholds can be evaluated against the untouched
                columns even after preprocessing has reshaped everything else.
            x_all: predictors for train and test stacked together.
            y_train: the target, indexed to match the training rows of ``x_all``.
        """
        index_col = self.cfg.index_column
        target = self.cfg.target_variable

        train = self._read(self.cfg.data.train_file)
        test = self._read(self.cfg.data.test_file)

        all_ids = pd.concat([train[index_col], test[index_col]], ignore_index=True)
        if not all_ids.is_unique:
            raise ValueError(f"'{index_col}' is not unique across train and test.")

        train = train.set_index(train[index_col] - 1).drop(index_col, axis=1)
        test = test.set_index(test[index_col] - 1).drop(index_col, axis=1)

        raw_train = train.copy()
        x_all = pd.concat([train.drop(target, axis=1), test], axis=0)
        y_train = train[target]

        self.logger.info(
            f"Loaded {train.shape[0]} train rows and {test.shape[0]} test rows "
            f"({x_all.shape[1]} predictors)"
        )
        return raw_train, x_all, y_train
