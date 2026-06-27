"""
Moduł trenowania modelu predykcji cen paliw (RandomForestRegressor).
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fuel_prices_processed.csv")
MODELS_DIR          = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILE_PATH     = os.path.join(MODELS_DIR, "fuel_price_model.pkl")

FEATURE_COLUMNS = [
    "brent_price_usd", "usd_pln_rate", "fuel_margin_pln",
    "month", "week_of_year", "day_of_week",
    "price_lag_1d", "price_lag_7d", "price_lag_14d",
    "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
]

TARGET_COLUMN           = "fuel_price_pln"
PREDICTION_HORIZON_DAYS = 14


def load_processed_data(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    logger.info("Wczytuję przetworzone dane z: %s", file_path)
    return pd.read_csv(file_path, parse_dates=["date"])


def build_training_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["target_price_14d"] = (
        dataframe.groupby("fuel_type")[TARGET_COLUMN].shift(-PREDICTION_HORIZON_DAYS)
    )
    return dataframe.dropna(subset=["target_price_14d"])


def get_available_features(dataframe: pd.DataFrame) -> list:
    available = [col for col in FEATURE_COLUMNS if col in dataframe.columns]
    missing   = [col for col in FEATURE_COLUMNS if col not in dataframe.columns]
    if missing:
        logger.warning("Brakujące cechy (pomijam): %s", missing)
    return available


def train_model(dataframe: pd.DataFrame, features: list) -> RandomForestRegressor:
    X = dataframe[features]
    y = dataframe["target_price_14d"]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        mae = mean_absolute_error(y.iloc[va], model.predict(X.iloc[va]))
        maes.append(mae)
        logger.info("Fold %d – MAE: %.4f PLN", fold, mae)

    logger.info("Średnie CV MAE: %.4f PLN", np.mean(maes))
    model.fit(X, y)
    return model


def evaluate_model(model, dataframe: pd.DataFrame, features: list) -> dict:
    split = int(len(dataframe) * 0.8)
    test  = dataframe.iloc[split:]
    preds = model.predict(test[features])
    actual = test["target_price_14d"]
    metrics = {
        "MAE":  round(mean_absolute_error(actual, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(actual, preds)), 4),
        "R2":   round(r2_score(actual, preds), 4),
    }
    logger.info("Metryki: %s", metrics)
    return metrics


def save_model(model, feature_names: list, output_path: str = MODEL_FILE_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, output_path)
    logger.info("Model zapisany do: %s", output_path)


def run_training_pipeline() -> dict:
    df       = load_processed_data()
    df       = build_training_target(df)
    features = get_available_features(df)
    model    = train_model(df, features)
    metrics  = evaluate_model(model, df, features)
    save_model(model, features)
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
