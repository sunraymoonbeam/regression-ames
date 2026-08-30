"""Prediction and Kaggle submission writing."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["Id", "SalePrice"]


def write_submission(predictions, index, cfg, name: str, logger: logging.Logger = None) -> Path:
    """Write a submission CSV in the format the competition expects.

    The frame is indexed by ``Id - 1``, so the Id column is the index plus one.
    """
    logger = logger or logging.getLogger(__name__)

    submission_dir = Path(cfg.data.submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.asarray(predictions, dtype=float).ravel()
    if np.isnan(predictions).any():
        raise ValueError(f"{name}: predictions contain NaN")
    if (predictions <= 0).any():
        raise ValueError(f"{name}: predictions contain non-positive prices")

    submission = pd.DataFrame(
        {SUBMISSION_COLUMNS[0]: np.asarray(index) + 1, SUBMISSION_COLUMNS[1]: predictions}
    )

    path = submission_dir / f"{name}_results_{cfg.training.target_transform}.csv"
    submission.to_csv(path, index=False)
    logger.info(
        f"Wrote {path} ({len(submission)} rows, "
        f"${predictions.min():,.0f}-${predictions.max():,.0f})"
    )
    return path


def save_model(model, cfg, name: str, logger: logging.Logger = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")
    return path


def load_model(path):
    return joblib.load(path)


def infer(model_path, X, cfg, name: str = "inference", logger: logging.Logger = None) -> Path:
    """Load a saved model and write predictions for X as a submission."""
    model = load_model(model_path)
    return write_submission(model.predict(X), X.index, cfg, name, logger)
