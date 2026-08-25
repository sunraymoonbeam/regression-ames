import sqlite3
import pandas as pd
import logging


class DatabaseManager:
    """
    Manages database connections and operations for SQLite databases.

    ## Attributes:
    --------------------------------
        * conn (sqlite3.Connection | None): The SQLite database connection object. Initially None until a connection is established.
        * logger (logging.Logger): Logger object for logging information, warnings, and errors.

    ## Methods:
    --------------------------------
        1. __init__(self, logger: logging.Logger) -> None:
            Initializes the DatabaseManager instance with a logger.

        2. create_connection(self, db_path: str) -> None:
            Establishes a SQLite database connection.

        3. query(self, sql_query: str) -> pd.DataFrame:
            Executes a SQL query and returns the results as a DataFrame.

        4. save_data(self, df: pd.DataFrame, output_filepath: str) -> None:
            Saves a DataFrame to a CSV file.

        5. close_connection(self) -> None:
            Closes the SQLite database connection.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the DatabaseManager instance with a logger.

        Args:
            logger (logging.Logger): Logger object for logging information, warnings, and errors.
        """
        self.conn = None
        self.df = None
        self.logger = logger

    def create_connection(self, db_path: str) -> None:
        """Establishes a SQLite database connection.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        try:
            self.conn = sqlite3.connect(db_path)
            self.logger.info(f"SQLite connection successfully established: {db_path}")
        except sqlite3.Error as e:
            self.logger.error(e)

    def query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the results as a DataFrame.

        Args:
            sql_query (str): SQL query to execute.

        Returns:
            pd.DataFrame: Query results.
        """
        self.df = pd.read_sql_query(sql_query, self.conn)

        self.df.to_csv("data/raw/raw.csv", index=False)
        self.logger.info(
            f"Query executed successfully: {sql_query}, data saved to data/raw/raw.csv"
        )
        return self.df

    def close_connection(self) -> None:
        """Closes the SQLite database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("SQLite connection is closed.")
