"""Ames housing price prediction -- end-to-end training pipeline.

Reproduces the approach developed in reports/notebooks/regression_ames_house_prices.ipynb
from the command line:

    python main.py
    python main.py training.models=[ridge,lasso] training.tune=false

The evaluation protocol is the point of this script as much as the models are.
Preprocessing is refit per fold, outliers leave the validation split alone,
models are ranked out-of-fold, blend weights come from out-of-fold predictions,
and the validation split is scored exactly once at the end.
"""

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.dataloader.dataloader import DataLoader
from src.features.cleaning import DataPreprocessor, outlier_index
from src.features.transform import FeatureTransformer
from src.pipe.sklearn_pipeline_manager import TransformerPipelineManager
from src.train.inference import save_model, write_submission
from src.train.model_factory import ModelFactory
from src.train.model_trainer import ModelTrainer, evaluate
from src.utils import Logger
from src.visualization.visualize import plot_feature_importance, plot_pipeline

logger = Logger(__name__, "logs/log.log").get_logger()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"{'-' * 20} Ames regression pipeline starting {'-' * 20}")
    pd.set_option("display.width", 120)

    # -- data ------------------------------------------------------------------
    raw_train, x_all, y_train = DataLoader(cfg, logger).load()

    # -- cleaning and feature engineering --------------------------------------
    x_all, features = DataPreprocessor(x_all, cfg, logger).preprocess()
    x_all, features = FeatureTransformer(x_all, features, cfg, logger).feature_engineer()
    outliers = outlier_index(raw_train, cfg, logger)

    # -- splits ----------------------------------------------------------------
    X = x_all.loc[raw_train.index]
    y = y_train.loc[raw_train.index]
    X_test = x_all.loc[x_all.index.difference(raw_train.index)]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=cfg.training.test_size, random_state=cfg.training.random_state
    )

    # Outliers come out of the training split only. Removing them before the split
    # also cleans the validation set, which flatters a score for something you
    # cannot do at inference time.
    before = len(X_train)
    keep = ~X_train.index.isin(outliers)
    X_train, Y_train = X_train.loc[keep], Y_train.loc[keep]
    logger.info(
        f"Train {before} -> {len(X_train)} rows after removing {before - len(X_train)} outliers | "
        f"val {len(X_val)} rows (untouched) | test {len(X_test)} rows"
    )

    # -- pipeline and models ---------------------------------------------------
    pipeline_manager = TransformerPipelineManager(features, cfg, logger)
    pipeline_manager.verify_coverage(x_all.columns)

    trainer = ModelTrainer(pipeline_manager, cfg, logger)
    models = ModelFactory(logger).get_models(list(cfg.training.models))
    fitted = trainer.fit_base_models(models, X_train, Y_train)

    if cfg.training.ensemble and len(fitted) > 1:
        base_estimators = trainer.best_estimators(fitted)
        logger.info("Training stacking regressor (out-of-fold meta-features)...")
        fitted["Stacking"] = trainer.build_stacking(base_estimators, X_train, Y_train)
        logger.info("Training voting regressor...")
        fitted["Voting"] = trainer.build_voting(base_estimators, X_train, Y_train)

    # -- evaluation ------------------------------------------------------------
    val_scores = {name: evaluate(Y_val, m.predict(X_val)) for name, m in fitted.items()}
    oof = trainer.out_of_fold(fitted, X_train, Y_train)
    comparison = trainer.comparison_table(oof, Y_train, val_scores)

    logger.info("Model comparison (ranked by out-of-fold RMSLE):")
    for line in comparison.to_string(index=False).splitlines():
        logger.info(f"  {line}")
    logger.info(
        "fold_std is the spread across folds -- where it exceeds the gap between "
        "models, the ranking is noise rather than a result."
    )

    # Selected on the out-of-fold score, not on the split being reported.
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    logger.info(
        f"Best model: {best_name} (OOF RMSLE {comparison.iloc[0]['oof_rmsle']:.5f} "
        f"+/- {comparison.iloc[0]['fold_std']:.5f}, val RMSLE {val_scores[best_name]['rmsle']:.5f})"
    )

    # -- submissions -----------------------------------------------------------
    write_submission(best_model.predict(X_test), X_test.index, cfg, best_name.lower(), logger)

    if cfg.training.blend and len(fitted) > 1:
        logger.info("Fitting blend weights on out-of-fold predictions...")
        names, weights = trainer.blend_weights(oof, Y_train)
        blend_val = trainer.blend_predict(fitted, X_val, names, weights)
        logger.info(f"Blend on the held-out validation split: {evaluate(Y_val, blend_val)}")
        write_submission(
            trainer.blend_predict(fitted, X_test, names, weights),
            X_test.index, cfg, "blend", logger,
        )

    # -- artefacts -------------------------------------------------------------
    if cfg.training.save_model:
        for name, model in fitted.items():
            save_model(model, cfg, name.lower(), logger)

    plot_pipeline(best_model, cfg, logger=logger)
    plot_feature_importance(best_model, cfg, logger=logger)

    logger.info(f"{'-' * 20} Ames regression pipeline complete {'-' * 20}")


if __name__ == "__main__":
    main()
