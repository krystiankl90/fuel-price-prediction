"""
Moduł ewaluacji modelu predykcji cen paliw (MAE, RMSE, R²).
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fuel_prices_processed.csv")
MODEL_FILE_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "fuel_price_model.pkl")
PREDICTION_HORIZON_DAYS = 14


def load_model_and_features(model_path: str = MODEL_FILE_PATH) -> tuple:
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_names"]


def compute_metrics(actual, predicted) -> dict:
    return {
        "MAE":  round(mean_absolute_error(actual, predicted), 4),
        "RMSE": round(np.sqrt(mean_squared_error(actual, predicted)), 4),
        "R2":   round(r2_score(actual, predicted), 4),
    }


def evaluate_per_fuel_type(dataframe, model, feature_names, test_fraction=0.2) -> dict:
    results = {}
    dataframe = dataframe.copy()
    dataframe["target_price_14d"] = (
        dataframe.groupby("fuel_type")["fuel_price_pln"].shift(-PREDICTION_HORIZON_DAYS)
    )
    dataframe = dataframe.dropna(subset=["target_price_14d"] + feature_names)

    for fuel_type, group in dataframe.groupby("fuel_type"):
        split = int(len(group) * (1 - test_fraction))
        test  = group.iloc[split:]
        if test.empty:
            continue
        preds   = model.predict(test[feature_names])
        metrics = compute_metrics(test["target_price_14d"], preds)
        results[fuel_type] = metrics
        logger.info("%s – MAE: %.4f, RMSE: %.4f, R2: %.4f", fuel_type, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    return results


def run_evaluation() -> dict:
    model, feature_names = load_model_and_features()
    dataframe = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
    results   = evaluate_per_fuel_type(dataframe, model, feature_names)

    print("\n=== Metryki modelu per rodzaj paliwa ===")
    for fuel_type, m in results.items():
        print(f"{fuel_type}: MAE={m['MAE']}, RMSE={m['RMSE']}, R2={m['R2']}")

    return results


if __name__ == "__main__":
    run_evaluation()
