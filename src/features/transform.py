import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from category_encoders import TargetEncoder
from typing import Any, Dict


class FeatureTransformer:
    """
    A class to transform raw data by performing feature engineering tasks such as generating new features,
    dropping unnecessary features, imputing missing values, removing outliers, scaling numerical data,
    and encoding categorical and ordinal data.

    ## Attributes:
    --------------------------------
        * df (DataFrame): Data to be preprocessed.
        * cfg (Dict): Configuration settings for data processing.
        * numerical_features (List[str]): List of numerical feature names.
        * categorical_features (List[str]): List of categorical feature names.
        * ordinal_features (List[str]): List of ordinal feature names.

        * numerical_imputer (SimpleImputer): Imputer for numerical features.
        * categorical_imputer (SimpleImputer): Imputer for categorical features.
        * ordinal_imputer (SimpleImputer): Imputer for ordinal features.

        * scaler (Union[StandardScaler, MinMaxScaler]): Scaler for numerical data.
        * encoder (Union[OneHotEncoder, TargetEncoder]): Encoder for categorical data.
        * ordinal_encoder (OrdinalEncoder): Encoder for ordinal data.

        * logger (Logger): Logger for logging events during data preprocessing.

    ## Methods:
    --------------------------------
        1. generate_new_features(): Generates new features from existing ones.
        2. drop_features(): Drops unnecessary features from the data.
        3. remove_outliers(): Removes outliers from the data.
        4. feature_engineer(): Main method to perform all feature engineering tasks.
        5. fit_transform(X_train, y_train): Fits the transformers and then transforms the data.
        6. transform(X_test, y_test): Transforms the data using the fitted transformers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Dict,
        logger: Any,
    ) -> None:
        """
        Initializes the FeatureTransformer with configuration settings.

        Args:
            logger: Logger for logging events during data preprocessing.
            cfg: Configuration settings for data processing.
        """
        self.df = df
        self.cfg = cfg

        # Initialize feature lists
        self.numerical_features = cfg["features"]["numerical"]
        self.categorical_features = cfg["features"]["categorical"]
        self.ordinal_features = cfg["features"]["ordinal"]
        self.features_to_drop = cfg["transformations"]["features_to_drop"]

        # Initialize imputers
        self.numerical_imputer = SimpleImputer(
            strategy=cfg["transformations"]["imputation"]["numerical"]
        )
        self.categorical_imputer = SimpleImputer(
            strategy=cfg["transformations"]["imputation"]["categorical"],
            fill_value="missing",
        )
        self.ordinal_imputer = SimpleImputer(
            strategy=cfg["transformations"]["imputation"]["ordinal"]
        )

        # Initialize scalers
        scaling_method = cfg["transformations"]["scaling_method"]
        self.scaler = (
            MinMaxScaler()
            if scaling_method == "minmax"
            else StandardScaler() if scaling_method == "standard" else None
        )

        # Initialize encoders
        encoding_method = cfg["transformations"]["categorical_encoding_method"]
        self.encoder = (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            if encoding_method == "onehot"
            else TargetEncoder() if encoding_method == "target" else None
        )

        self.ordinal_encoder = OrdinalEncoder()

        self.remove_outliers_true = cfg["transformations"]["remove_outliers"]
        self.logger = logger

    def generate_new_features(self) -> None:
        """
        Generates new features from existing ones.

        Args:
            df: DataFrame to generate new features for.

        Returns:
            DataFrame with new features added.
        """

        # Track initial feature count for logging
        initial_feature_count = self.df.shape[1]

        if "last_weight" in self.df.columns and "current_weight" in self.df.columns:
            self.df["weight_change"] = (
                self.df["last_weight"] - self.df["current_weight"]
            )
            self.df["weight_loss_true"] = np.where(
                self.df["weight_change"] < 0, "Yes", "No"
            )
            self.numerical_features.append("weight_change")
            self.categorical_features.append("weight_loss_true")

        # Log the generated features
        new_feature_count = self.df.shape[1] - initial_feature_count
        new_features = self.df.columns[-new_feature_count:]
        self.logger.info(
            f"Generated {new_feature_count} new features: {list(new_features)}"
        )

    def drop_features(self):
        """
        Drops unnecessary features from the DataFrame.

        Returns:
            DataFrame with specified features dropped.
        """
        initial_feature_count = self.df.shape[1]

        self.df.drop(columns=self.features_to_drop, inplace=True)

        # Update feature lists after dropping columns
        self.numerical_features = [
            feature
            for feature in self.numerical_features
            if feature not in self.features_to_drop
        ]
        self.categorical_features = [
            feature
            for feature in self.categorical_features
            if feature not in self.features_to_drop
        ]
        self.ordinal_features = [
            feature
            for feature in self.ordinal_features
            if feature not in self.features_to_drop
        ]

        # Log the dropped features
        dropped_feature_count = initial_feature_count - self.df.shape[1]
        self.logger.info(
            f"Dropped {dropped_feature_count} features: {self.features_to_drop}"
        )

    def remove_outliers(self):
        """
        Removes outliers from the DataFrame using the interquartile range method.

        Args:
            df: DataFrame from which to remove outliers.
            numerical_features: List of numerical feature names to check for outliers.

        Returns:
            DataFrame with outliers removed.
        """
        initial_row_count = self.df.shape[0]

        for col in self.numerical_features:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[
                ~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))
            ]

        removed_outliers_count = initial_row_count - self.df.shape[0]
        self.logger.info(f"Removed {removed_outliers_count} outliers.")

    def feature_engineer(
        self,
    ) -> pd.DataFrame:
        """
        Main method to perform all feature engineering tasks on the given DataFrame.

        Args:

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """

        if self.remove_outliers_true:
            self.remove_outliers()
        self.generate_new_features()
        self.drop_features()

        # Ensure no null values remain
        assert not self.df.isnull().any().any(), "Null values found after processing."

        self.logger.info(f"Feature Engineering completed.")
        return self.df

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Fits the transformers to the training data and transforms the training data.

        Args:
            X_train: Training data DataFrame.

        Returns:
            Transformed training data DataFrame.
        """
        # Impute missing values
        if self.cfg.transformations.impute:
            X_train[self.numerical_features] = self.numerical_imputer.fit_transform(
                X_train[self.numerical_features]
            )
            X_train[self.categorical_features] = self.categorical_imputer.fit_transform(
                X_train[self.categorical_features]
            )
            X_train[self.ordinal_features] = self.ordinal_imputer.fit_transform(
                X_train[self.ordinal_features]
            )

        # Scale numerical features
        if self.cfg.transformations.scaling:
            X_train[self.numerical_features] = self.scaler.fit_transform(
                X_train[self.numerical_features]
            )

        # Encode categorical features
        if isinstance(self.encoder, OneHotEncoder):
            X_train = pd.concat(
                [
                    X_train.drop(columns=self.categorical_features),
                    pd.DataFrame(
                        self.encoder.fit_transform(X_train[self.categorical_features]),
                        columns=self.encoder.get_feature_names_out(),
                        index=X_train.index,
                    ),
                ],
                axis=1,
            )
        elif isinstance(self.encoder, TargetEncoder):
            X_train[self.categorical_features] = self.encoder.fit_transform(
                X_train[self.categorical_features],
                y_train,
            )

        # Encode ordinal features
        X_train[self.ordinal_features] = self.ordinal_encoder.fit_transform(
            X_train[self.ordinal_features]
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        train_df.to_csv("data/processed/train_data.csv", index=False)
        self.logger.info(
            "Feature Transformation: Training data transformed successfully."
        )
        return X_train, y_train

    def transform(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Fits the transformers to the training data and transforms the training data.

        Args:
            X_test: Test data DataFrame.

        Returns:
            Transformed test data DataFrame.
        """
        # Impute missing values
        X_test[self.numerical_features] = self.numerical_imputer.fit_transform(
            X_test[self.numerical_features]
        )
        X_test[self.categorical_features] = self.categorical_imputer.fit_transform(
            X_test[self.categorical_features]
        )
        X_test[self.ordinal_features] = self.ordinal_imputer.fit_transform(
            X_test[self.ordinal_features]
        )

        # Scale numerical features
        if self.cfg.transformations.scaling and self.scaler is not None:
            X_test[self.numerical_features] = self.scaler.fit_transform(
                X_test[self.numerical_features]
            )

        # Encode categorical features
        if isinstance(self.encoder, OneHotEncoder):
            X_test = pd.concat(
                [
                    X_test.drop(columns=self.categorical_features),
                    pd.DataFrame(
                        self.encoder.fit_transform(X_test[self.categorical_features]),
                        columns=self.encoder.get_feature_names_out(),
                        index=X_test.index,
                    ),
                ],
                axis=1,
            )
        elif isinstance(self.encoder, TargetEncoder):
            X_test[self.categorical_features] = self.encoder.fit_transform(
                X_test[self.categorical_features],
                y_test,
            )

        # Encode ordinal features
        X_test[self.ordinal_features] = self.ordinal_encoder.fit_transform(
            X_test[self.ordinal_features]
        )

        test_df = pd.concat([X_test, y_test], axis=1)
        test_df.to_csv("data/processed/test_data.csv", index=False)
        self.logger.info("Feature Transformation: Test data transformed successfully.")
        return X_test, y_test
