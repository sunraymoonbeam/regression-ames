"""
This script makes predictions using a trained machine learning model.

The script follows these steps:
1. Loads data from the input file.
2. Preprocesses and transforms the data.
3. Loads a model from the model file.
4. Makes predictions on the processed data.
5. Saves the predictions to the output file.

Example usage:
    python inference.py --input_filepath ./data/raw/raw.csv --model_filepath ./models/model.pkl --output_filepath ./predictions/predictions.csv
"""

import argparse
import logging
import pandas as pd
import pickle
from features.cleaning import DataPreprocessor
from src.features.transform import FeatureTransformer


def infer(args):
    """
    Loads data from the input file, preprocesses and transforms it,
    loads a model from the model file, makes predictions on the processed data,
    and saves the predictions to the output file.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.
    """
    # Set up logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Load the data
    logger.info(f"Loading data from {args.input_filepath}")
    df = pd.read_csv(args.input_filepath)

    # Preprocess the data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.preprocess(df)

    # Transform the data
    logger.info("Transforming data")
    transformer = FeatureTransformer()
    processed_df = transformer.transform(cleaned_df)

    # Load the model
    logger.info(f"Loading model from {args.model_filepath}")
    with open(args.model_filepath, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(processed_df)

    # Save predictions to a CSV file
    logger.info(f"Saving predictions to {args.output_filepath}")
    pd.DataFrame(predictions).to_csv(args.output_filepath, index=False)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model."
    )
    parser.add_argument(
        "input_filepath",
        type=str,
        help="Path to the input data file. The file should be in CSV format.",
    )
    parser.add_argument(
        "model_filepath",
        type=str,
        help="Path to the trained model file. The model should be a pickled object.",
    )
    parser.add_argument(
        "output_filepath",
        type=str,
        help="Path to save the predictions. The predictions will be saved in CSV format.",
    )
    args = parser.parse_args()

    infer(args)
