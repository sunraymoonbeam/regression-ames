import logging
import requests
import os


class DataLoader:
    """
    Handles the downloading of data from a specified URL.

    ## Attributes:
        * logger (logging.Logger): Logger object for logging information, warnings, and errors.
    --------------------------------

    ## Methods:
    --------------------------------
        1. __init__(self, logger: logging.Logger) -> None:
            Initializes the DataLoader instance with a logger.

        2. download(self, url: str, save_filepath: str = "src/data/external/lung_cancer.db") -> str:
            Downloads a file from a given URL to a specified local file path.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the DataLoader instance with a logger.

        Args:
            logger (logging.Logger): Logger object for logging information, warnings, and errors.
        """
        self.logger = logger

    def download(
        self, url: str, save_filepath: str = "data/external/lung_cancer.db"
    ) -> str:
        """
        Downloads a file from a given URL to a specified local file path.

        Args:
            url (str): URL to download the file from.
            save_filepath (str): Local path to save the downloaded file. Defaults to "data/external/lung_cancer.db".

        Returns:
            str: The file path where the downloaded file is saved.
        """
        response = requests.get(url)
        with open(save_filepath, "wb") as f:
            f.write(response.content)
        self.logger.info(
            f"Database file successfully downloaded from {url} to {save_filepath}"
        )
        return save_filepath
