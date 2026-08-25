# Import configurations and utilities
import hydra
from omegaconf import DictConfig
from conf.model_config import (
    DEFAULT_MODEL_PARAMETERS,
    MODEL_PARAM_GRIDS,
    MODEL_NAME_MAPPING,
)

# Import custom modules
from src.dataloader.dataloader import DataLoader
from src.dataloader.database_manager import DatabaseManager
from src.features.cleaning import DataPreprocessor
from src.features.transform import FeatureTransformer
from src.train.model_factory import ModelFactory
from src.pipe.sklearn_pipeline_manager import TransformerPipelineManager
from sklearn.model_selection import train_test_split
from src.visualization.visualize import (
    plot_feature_importance_model,
    plot_feature_importance_pipeline,
    plot_pipeline,
)
from src.utils import Logger

# Initialize logger
logger = Logger(__name__, "logs/sklearn_pipeline.log").get_logger()
logger.info(f"{'-'*25} Starting the Machine learning pipeline {'-'*25}".center(50))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to orchestrate the machine learning pipeline from data loading,
    preprocessing, training, and evaluation.
    """
    # Download and load data
    data_loader = DataLoader(logger=logger)
    save_path = data_loader.download(cfg.download_url)

    db_manager = DatabaseManager(logger=logger)
    db_manager.create_connection(save_path)
    df = db_manager.query("SELECT * FROM lung_cancer")
    db_manager.close_connection()

    # Data preprocessing
    preprocessor = DataPreprocessor(df=df, cfg=cfg, logger=logger)
    cleaned_df = preprocessor.preprocess()

    # Feature Engineering
    transformer = FeatureTransformer(df=cleaned_df, cfg=cfg, logger=logger)
    processed_df = transformer.feature_engineer()

    # Data splitting (Avoid data leakage by splitting before feature transformation)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_df.drop(cfg.features.target_variable, axis=1),
        processed_df[cfg.features.target_variable],
        test_size=0.2,
        random_state=42,
    )

    # Model training and evaluation
    model_factory = ModelFactory()
    mapped_models = [
        MODEL_NAME_MAPPING[model]
        for model in cfg.training.models
        if model in MODEL_NAME_MAPPING
    ]
    models = model_factory.get_models(mapped_models if mapped_models else ["all"])

    # Train and evaluate models
    pipeline_manager = TransformerPipelineManager(logger=logger)
    transformations = pipeline_manager.create_transformations(
        cfg.transformations.scaling_method,
        cfg.transformations.categorical_encoding_method,
        numeric_columns=transformer.numerical_features,
        categorical_columns=transformer.categorical_features,
        ordinal_columns=transformer.ordinal_features,
    )
    pipeline_manager.create_pipeline()
    pipeline_manager.create_param_grid(transformations, models, MODEL_PARAM_GRIDS)
    pipeline_manager.fit_pipeline(
        X_train,
        y_train,
        n_iter=cfg.training.n_iter,
        tune_metric=cfg.training.tune_metric,
    )
    pipeline_manager.evaluate_pipeline(X_test, y_test)
    plot_pipeline(
        pipeline_manager.best_model_pipeline.best_estimator_,
        file_name="best_pipeline",
        save_dir="reports/figures/",
    )
    plot_feature_importance_pipeline(
        pipeline_manager.best_model_pipeline.best_estimator_,
        "reports/figures/best_pipeline_feature_importance.png",
    )

    logger.info("Machine Learning Pipeline Completed".center(50, "-"))


if __name__ == "__main__":
    main()
