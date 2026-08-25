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
from src.train.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from src.utils import Logger

# Initialize logger
logger = Logger(__name__, "logs/log.log").get_logger()
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

    # Feature transformation
    X_train, y_train = transformer.fit_transform(X_train, y_train)
    X_test, y_test = transformer.transform(X_test, y_test)

    # Model training and evaluation
    model_factory = ModelFactory()
    mapped_models = [
        MODEL_NAME_MAPPING[model]
        for model in cfg.training.models
        if model in MODEL_NAME_MAPPING
    ]
    models = model_factory.get_models(mapped_models if mapped_models else ["all"])

    results = []
    for model_name, model in models.items():
        trainer = ModelTrainer(
            model,
            parameters={
                **DEFAULT_MODEL_PARAMETERS[model_name],
                "verbose": cfg.training.verbose,
            },
            logger=logger,
        )
        if cfg.training.tune:
            param_grid = MODEL_PARAM_GRIDS[model_name]
            best_params = trainer.random_tune_parameters(
                X_train,
                y_train,
                param_grid,
                tune_metric=cfg.training.tune_metric,
                verbose=cfg.training.verbose,
                n_iter=cfg.training.n_iter,
            )
            trainer.set_params(best_params)
        trainer.train(X_train, y_train)
        models[model_name] = trainer.model

        y_pred = trainer.predict(X_test)
        result = trainer.evaluate(y_test, y_pred)
        result["model_name"] = model_name
        results.append(result)
        if cfg.training.save_model:
            model_save_path = f"{cfg.training.save_dir}/{model_name}.pkl"
            trainer.save_model(model_save_path)

    if cfg.training.ensemble and len(models) > 1:
        ensemble_models = [(name, model) for name, model in models.items()]
        trainer = ModelTrainer(None, parameters={}, logger=logger)

        # Voting Classifier
        trainer.model = trainer.train_voting_classifier(
            ensemble_models, X_train, y_train
        )
        voting_results = trainer.evaluate(y_test, trainer.model.predict(X_test))
        voting_results["model_name"] = "VotingRegressor"
        results.append(voting_results)
        trainer.save_model(f"{cfg.training.save_dir}/VotingRegressor.pkl")

        # Stacking Classifier
        final_estimator = ModelFactory().get_models(["LogisticRegression"])[
            "LogisticRegression"
        ]
        trainer.model = trainer.train_stacking_classifier(
            ensemble_models, final_estimator, X_train, y_train
        )
        stacking_results = trainer.evaluate(y_test, trainer.model.predict(X_test))
        stacking_results["model_name"] = "StackingRegressor"
        results.append(stacking_results)
        trainer.save_model(f"{cfg.training.save_dir}/StackingRegressor.pkl")

    results.sort(key=lambda x: x[cfg.training.performance_metric], reverse=True)
    best_result = results[0]
    performance_metrics = ", ".join(
        f"{metric}: {score:.2f}"
        for metric, score in best_result.items()
        if metric != "model_name"
    )
    logger.info(
        f"Best model ranked using {cfg.training.performance_metric}: {best_result['model_name']} with performance metrics: {performance_metrics}"
    )
    logger.info(f"{'-'*25} Machine learning pipeline completed {'-'*25}".center(50))


if __name__ == "__main__":
    main()
