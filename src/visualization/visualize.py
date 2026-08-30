"""Plots for the trained pipeline."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.utils import estimator_html_repr


def _figures_dir(cfg) -> Path:
    path = Path("reports/figures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_pipeline(estimator, cfg, name: str = "best_pipeline", logger: logging.Logger = None) -> Path:
    """Write the pipeline diagram to a standalone HTML file."""
    logger = logger or logging.getLogger(__name__)
    set_config(display="diagram")
    path = _figures_dir(cfg) / f"{name}.html"
    path.write_text(estimator_html_repr(estimator), encoding="utf-8")
    logger.info(f"Wrote pipeline diagram to {path}")
    return path


def plot_feature_importance(estimator, cfg, name: str = "best_pipeline_feature_importance",
                            top_n: int = 30, logger: logging.Logger = None):
    """Bar chart of feature importances or absolute coefficients, if available.

    Returns None for models that expose neither (a voting regressor, say), rather
    than inventing an importance that does not exist.
    """
    logger = logger or logging.getLogger(__name__)

    pipeline = getattr(estimator, "regressor_", estimator)
    pipeline = getattr(pipeline, "best_estimator_", pipeline)
    try:
        model = pipeline.steps[-1][1]
        preprocessor = pipeline.named_steps["columntransformer"]
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, KeyError):
        logger.warning("Could not locate the model or feature names; skipping importance plot")
        return None

    if hasattr(model, "feature_importances_"):
        values, label = np.asarray(model.feature_importances_), "importance"
    elif hasattr(model, "coef_"):
        values, label = np.abs(np.ravel(model.coef_)), "|coefficient|"
    else:
        logger.info(f"{type(model).__name__} exposes no importances; skipping plot")
        return None

    if len(values) != len(feature_names):
        logger.warning("Importance/feature-name length mismatch; skipping importance plot")
        return None

    series = pd.Series(values, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(series))))
    series.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel(label)
    ax.set_title(f"Top {len(series)} features ({type(model).__name__})")
    fig.tight_layout()

    path = _figures_dir(cfg) / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Wrote feature importance plot to {path}")
    return path
