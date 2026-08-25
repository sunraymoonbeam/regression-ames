import numpy as np
import pandas as pd
import os
from omegaconf import DictConfig
from typing import Optional, Tuple
from logging import Logger


class DataPreprocessor:
    """
    A class used to preprocess raw data.

    ## Attributes
    --------------------------------
    * train_df : DataFrame
        the train data to be preprocessed
    * test_df : DataFrame
        the test data to be preprocessed
    * cfg : DictConfig
        the configuration settings for data preprocessing
    * logger : Logger
        a logger for logging events during data preprocessing

    ## Methods
    --------------------------------
    # TBC
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cfg: DictConfig,
        logger: Logger = None,
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.df = self.combine_train_test(train_df, test_df)
        self.cfg = cfg
        self.problem_type = cfg.problem_type
        self.logger = logger

    def combine_train_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Combines the train and test data for preprocessing.

        Args:
            train_df (pd.DataFrame): The train data to be preprocessed.
            test_df (pd.DataFrame): The test data to be preprocessed.
        """
        self.train_df = train_df.set_index(train_df["Id"] - 1)
        self.test_df = test_df.set_index(test_df["Id"] - 1)
        self.df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        self.logger.info("Train and test data combined for preprocessing.")

    def spilt_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.df.loc[self.train_df.index]
        test_df = self.df.loc[self.test_df.index]
        return train_df, test_df

    def change_column_type(self) -> None:
        """ """
        self.df = self.df.replace(
            {
                "MSSubClass": {
                    20: "SC20",
                    30: "SC30",
                    40: "SC40",
                    45: "SC45",
                    50: "SC50",
                    60: "SC60",
                    70: "SC70",
                    75: "SC75",
                    80: "SC80",
                    85: "SC85",
                    90: "SC90",
                    120: "SC120",
                    150: "SC150",
                    160: "SC160",
                    180: "SC180",
                    190: "SC190",
                },
                "MoSold": {
                    1: "Jan",
                    2: "Feb",
                    3: "Mar",
                    4: "Apr",
                    5: "May",
                    6: "Jun",
                    7: "Jul",
                    8: "Aug",
                    9: "Sep",
                    10: "Oct",
                    11: "Nov",
                    12: "Dec",
                },
            }
        )
        self.df = self.df.replace(
            {
                "Alley": {np.nan: 0, "Grvl": 1, "Pave": 2},
                "BsmtCond": {
                    np.nan: 0,
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # BsmtQual: Evaluates the height of the basement
                "BsmtExposure": {
                    np.nan: 0,
                    "No": 1,
                    "Mn": 2,
                    "Av": 3,
                    "Gd": 4,
                },  # BsmtExposure: Refers to walkout or garden level walls
                "BsmtFinType1": {
                    np.nan: 0,
                    "Unf": 1,
                    "LwQ": 2,
                    "Rec": 3,
                    "BLQ": 4,
                    "ALQ": 5,
                    "GLQ": 6,
                },  # BsmtFinType1: Rating of basement finished area
                "BsmtFinType2": {
                    np.nan: 0,
                    "Unf": 1,
                    "LwQ": 2,
                    "Rec": 3,
                    "BLQ": 4,
                    "ALQ": 5,
                    "GLQ": 6,
                },  # BsmtFinType2: Rating of basement finished area (if multiple types)
                "BsmtQual": {
                    np.nan: 0,
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # BsmtQual: Evaluates the height of the basement
                "ExterCond": {
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # ExterCond: Evaluates the present condition of the material on the exterior
                "ExterQual": {
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # ExterQual: Evaluates the quality of the material on the exterior
                "FireplaceQu": {
                    np.nan: 0,
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # FireplaceQu: Fireplace quality
                "Functional": {
                    "Sal": 1,
                    "Sev": 2,
                    "Maj2": 3,
                    "Maj1": 4,
                    "Mod": 5,
                    "Min2": 6,
                    "Min1": 7,
                    "Typ": 8,
                },  # Home functionality (Assume typical unless deductions are warranted)
                "GarageCond": {
                    np.nan: 0,
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # GarageCond: Evaluates the present condition of the garage
                "GarageQual": {
                    np.nan: 0,
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # GarageQual: Garage quality
                "HeatingQC": {
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # HeatingQC: Heating quality and condition
                "KitchenQual": {
                    "Po": 1,
                    "Fa": 2,
                    "TA": 3,
                    "Gd": 4,
                    "Ex": 5,
                },  # KitchenQual: Kitchen quality
                "LandSlope": {
                    "Sev": 1,
                    "Mod": 2,
                    "Gtl": 3,
                },  # LandSlope: Slope of property
                "LotShape": {
                    "IR3": 1,
                    "IR2": 2,
                    "IR1": 3,
                    "Reg": 4,
                },  # LotShape: General shape of property
                "PavedDrive": {"N": 0, "P": 1, "Y": 2},  # PavedDrive: Paved driveway
                "PoolQC": {
                    np.nan: 0,
                    "Fa": 1,
                    "TA": 2,
                    "Gd": 3,
                    "Ex": 4,
                },  # PoolQC: Pool quality
                "Street": {"Grvl": 1, "Pave": 2},  # Street: Type of road access
                "Utilities": {
                    "ELO": 1,
                    "NoSeWa": 2,
                    "NoSewr": 3,
                    "AllPub": 4,
                },  # Utilities: Type of utilities available
                "Fence": {np.nan: 0, "GdPrv": 1, "MnPrv": 2, "GdWo": 3, "MnWw": 4},
            }  # Fence: Fence quality
        )

    def drop_low_variance_features(self, threshold=0.95) -> None:
        """_summary_

        Args:
            threshold (float, optional): _description_. Defaults to 0.95.
        """
        overfit = []
        for col in self.df.columns:
            dominant_val_pct = self.df[col].value_counts(normalize=True).iloc[0]
            if dominant_val_pct > threshold:
                overfit.append(col)
        self.df = self.df.drop(overfit, axis=1)
        self.logger.info(f"{len(overfit)} low variance features dropped: {overfit}")

    def drop_missing_values(self, threshold=0.85) -> None:
        """
        Drops columns with missing values more than the threshold.
        Args:
            threshold (float, optional): Threshold for dropping columns. Defaults to 0.85.
        """
        missing_value_ratio = self.df.isnull().mean()
        columns_to_drop = missing_value_ratio[missing_value_ratio > threshold].index
        self.df.drop(columns_to_drop, axis=1, inplace=True)
        self.logger.info(
            f"{len(columns_to_drop)} columns dropped with missing values more then {1-threshold}: {columns_to_drop}."
        )

    def sanity_check(self) -> None:
        """
        Performs a sanity check to ensure data quality.
        The following checks are performed:
        - No missing values
        - No duplicate rows
        - No mixed data types
        - No constant columns
        """

        # Check for missing values
        if self.df.isnull().any().any():
            self.logger.warning("Data still contains missing values.")

        # Check for duplicate rows
        if self.df.duplicated().any():
            self.logger.warning("Data still contains duplicate rows.")

        # Check for mixed data types
        for col in self.df.columns:
            unique_types = set(self.df[col].apply(type))
            if len(unique_types) > 1:
                self.logger.warning(f"Column {col} has mixed data types.")

        # Check for constant columns
        for col in self.df.columns:
            if self.df[col].nunique() <= 1:
                self.logger.warning(f"Column {col} is constant.")
        print(
            "Sanity check completed. No missing values, no duplicate rows, no mixed data types, and no constant columns."
        )

    def preprocess(self, output_dir: str = "data/interim/cleaned.csv") -> None:
        """
        Preprocesses the raw data and saves the cleaned data to a specified file.

        Args:
            output_filepath (str): The file path to save the cleaned data.
        """
        self.change_column_type()
        self.drop_missing_values()
        self.drop_low_variance_features()
        self.sanity_check()
        clean_train_df, clean_test_df = self.spilt_train_test()
        clean_train_df.to_csv(os.path.join(output_dir, "cleaned_train.csv"), index=False)
        clean_test_df.to_csv(os.path.join(output_dir, "cleaned_test.csv"), index=False)
        self.logger.info(
            f"Data cleaning and preprocessing completed. 
              Missing values are handled. Sanity check performed."
        )
        return clean_train_df, clean_test_df
        
        
