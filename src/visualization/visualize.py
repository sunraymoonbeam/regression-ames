import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from html2image import Html2Image
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from sklearn import set_config

set_config(
    display="diagram"
)  # Set config to 'diagram' so we can visualize pipelines/composite estimators


def plot_feature_importance_model(
    model: BaseEstimator,
    feature_names: Union[list, pd.Index],
    filepath: Optional[str] = None,
) -> None:
    """
    Plots the feature importances of a fitted model and saves the plot to a file.

    Args:
        model (BaseEstimator): The fitted model with feature importances or coefficients.
        feature_names (Union[list, pd.Index]): The names of the features corresponding to the importances or coefficients.
        filepath (Optional[str], optional): The path to save the plot. If None, a default path is generated. Defaults to None.
    """
    # Generate default filepath if not provided
    if filepath is None:
        model_name = type(model).__name__
        filepath = f"reports/figures/{model_name}_feature_importance.jpg"

    # Determine the type of importance values the model provides
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        print(
            f"The model {type(model).__name__} does not support feature importances or coefficients."
        )
        return

    # Create a series for the feature importances
    feat_importances = pd.Series(importances, index=feature_names)

    # Plotting
    plt.figure(figsize=(10, 8))
    feat_importances.nlargest(10).plot(kind="barh")
    plt.title("Feature Importances")
    plt.savefig(filepath)
    plt.close()


def plot_pipeline(
    pipeline: Pipeline,
    file_name: str = "best_pipeline",
    save_dir: str = "reports/figures/",
) -> None:
    """
    Visualizes a scikit-learn pipeline and saves it as both HTML and PNG formats.

    Args:
        pipeline (Pipeline): The scikit-learn pipeline to visualize.
        file_name (str, optional): The base name of the file to save the visualization as. Defaults to "best_pipeline".
        save_dir (str, optional): The directory to save the visualization files in. Defaults to "reports/figures/".
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate HTML representation of the pipeline
    html_repr = estimator_html_repr(pipeline)

    # Save the HTML representation to a file
    html_file_path = os.path.join(save_dir, f"{file_name}.html")
    with open(html_file_path, "w") as html_file:
        html_file.write(html_repr)

    # Initialize Html2Image with specified paths
    hti = Html2Image(temp_path=save_dir, output_path=save_dir)

    # Convert HTML to PNG and save
    hti.screenshot(html_str=html_repr, save_as=f"{file_name}.png", size=(800, 600))


def plot_feature_importance_pipeline(
    pipeline: Pipeline, filepath: Optional[str] = None
) -> None:
    """
    Plots the feature importances of the final estimator in a pipeline.

    Args:
        pipeline (Pipeline): The scikit-learn pipeline with a final estimator that supports feature importances or coefficients.
        filepath (Optional[str], optional): The path to save the plot. If None, a default path is generated. Defaults to None.
    """
    # Extract the final model and transformer from the pipeline
    model = pipeline.named_steps[
        "clf"
    ]  # Adjust 'clf' to your final estimator's key in the pipeline
    transformer = pipeline.named_steps[
        "data_transformations"
    ]  # Adjust to your transformer's key

    # Get transformed feature names
    feature_names = transformer.get_feature_names_out()

    # Plot feature importances for the model
    plot_feature_importance_model(model, feature_names, filepath)
