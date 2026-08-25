import logging


class Logger:
    """A custom logger class that simplifies the use of logging.

    Attributes:
        logger (Logger): An instance of the Logger class.
    """

    def __init__(self, name, log_file, level=logging.INFO):
        """Initializes the Logger class with a name, log file and level.

        Args:
            name (str): The name of the logger.
            log_file (str): The file to which the log will be written.
            level (int, optional): The level of the logger. Defaults to logging.INFO.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M",
        )

        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create and configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """Returns the logger instance.

        Returns:
            Logger: The logger instance.
        """
        return self.logger
