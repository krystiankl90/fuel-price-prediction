"""
Moduł przygotowania danych do modelu predykcji cen paliw.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")
RAW_FILE_PATH      = os.path.join(RAW_DATA_DIR, "fuel_prices.csv")
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "fuel_prices_processed.csv")

COVID_YEARS = [2020, 2021]


def load_raw_data(file_path: str) -> pd.DataFrame:
    logger.info("Wczytuję dane z: %s", file_path)
    dataframe = pd.read_csv(file_path, parse_dates=["date"])
    logger.info("Wczytano %d wierszy.", len(dataframe))
    return dataframe


def remove_covid_period(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(dataframe)
    dataframe = dataframe[~dataframe["date"].dt.year.isin(COVID_YEARS)].copy()
    logger.info("Usunięto %d wierszy z okresu COVID (%s).", rows_before - len(dataframe), COVID_YEARS)
    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop_duplicates(subset=["date", "fuel_type"])
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].interpolate(method="linear")
    dataframe = dataframe.dropna()
    logger.info("Po czyszczeniu zostało %d wierszy.", len(dataframe))
    return dataframe


def add_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["month"]       = dataframe["date"].dt.month
    dataframe["week_of_year"] = dataframe["date"].dt.isocalendar().week.astype(int)
    dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
    return dataframe


def add_lag_features(dataframe: pd.DataFrame, lag_days: list = None) -> pd.DataFrame:
    if lag_days is None:
        lag_days = [1, 7, 14]
    for lag in lag_days:
        dataframe[f"price_lag_{lag}d"] = dataframe.groupby("fuel_type")["fuel_price_pln"].shift(lag)
        if "brent_price_usd" in dataframe.columns:
            dataframe[f"brent_lag_{lag}d"] = dataframe.groupby("fuel_type")["brent_price_usd"].shift(lag)
    return dataframe


def add_rolling_features(dataframe: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 30]
    for window in windows:
        dataframe[f"rolling_mean_{window}d"] = (
            dataframe.groupby("fuel_type")["fuel_price_pln"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return dataframe


def prepare_data(input_path: str = RAW_FILE_PATH, output_path: str = PROCESSED_FILE_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe = load_raw_data(input_path)
    dataframe = remove_covid_period(dataframe)
    dataframe = clean_data(dataframe)
    dataframe = add_time_features(dataframe)
    dataframe = add_lag_features(dataframe)
    dataframe = add_rolling_features(dataframe)
    dataframe = dataframe.dropna().reset_index(drop=True)
    dataframe.to_csv(output_path, index=False)
    logger.info("Zapisano przetworzone dane do: %s", output_path)
    return dataframe


if __name__ == "__main__":
    prepare_data()
