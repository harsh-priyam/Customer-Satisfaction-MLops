import logging
import pandas as pd
from zenml import step 

from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

from src.data_cleaning import(
    DataCleaning,
    DataDivideStrategy,
    DataPreProcessStrategy,
)


@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divides it into train and test

    Args:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing Labels
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
