"""
Moduł predykcji ceny paliwa za 14 dni.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fuel_prices_processed.csv")
MODEL_FILE_PATH         = os.path.join(os.path.dirname(__file__), "..", "models", "fuel_price_model.pkl")
PRICE_CHANGE_THRESHOLD  = 0.05


def load_model_bundle(model_path: str = MODEL_FILE_PATH) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}. Uruchom najpierw: python main.py")
    bundle = joblib.load(model_path)
    logger.info("Wczytano model z: %s", model_path)
    return bundle


def load_latest_fuel_data(fuel_type: str, data_path: str = PROCESSED_DATA_PATH) -> pd.Series:
    df = pd.read_csv(data_path, parse_dates=["date"])
    fuel_data = df[df["fuel_type"] == fuel_type].sort_values("date")
    if fuel_data.empty:
        raise ValueError(f"Brak danych dla paliwa: {fuel_type}")
    return fuel_data.iloc[-1]


def determine_price_direction(current_price: float, predicted_price: float) -> str:
    diff = predicted_price - current_price
    if diff > PRICE_CHANGE_THRESHOLD:
        return "wzrost"
    if diff < -PRICE_CHANGE_THRESHOLD:
        return "spadek"
    return "stabilnie"


def build_prediction_justification(fuel_type, latest_row, predicted_price, direction) -> str:
    brent  = latest_row.get("brent_price_usd", None)
    usd    = latest_row.get("usd_pln_rate", None)
    brent_info = f"Cena ropy Brent: {brent:.1f} USD/bbl. " if brent else ""
    usd_info   = f"Kurs USD/PLN: {usd:.4f}. " if usd else ""
    texts = {
        "wzrost":    "Model przewiduje wzrost ceny – możliwe przyczyny: rosnąca ropa lub osłabienie złotego.",
        "spadek":    "Model przewiduje spadek ceny – możliwe przyczyny: tańsza ropa lub umocnienie złotego.",
        "stabilnie": "Model nie przewiduje istotnych zmian ceny w ciągu najbliższych 14 dni.",
    }
    return f"{brent_info}{usd_info}{texts.get(direction, '')}"


def predict_fuel_price(fuel_type: str) -> dict:
    bundle   = load_model_bundle()
    model    = bundle["model"]
    features = bundle["feature_names"]

    latest_row    = load_latest_fuel_data(fuel_type)
    current_price = latest_row["fuel_price_pln"]

    feature_vector  = pd.DataFrame([[latest_row.get(col, np.nan) for col in features]], columns=features)
    predicted_price = round(float(model.predict(feature_vector)[0]), 3)
    price_diff      = round(predicted_price - current_price, 3)
    direction       = determine_price_direction(current_price, predicted_price)
    justification   = build_prediction_justification(fuel_type, latest_row, predicted_price, direction)

    return {
        "fuel_type":      fuel_type,
        "current_price":  round(current_price, 3),
        "predicted_price": predicted_price,
        "price_difference": price_diff,
        "direction":      direction,
        "justification":  justification,
        "last_data_date": str(latest_row["date"])[:10],
    }


if __name__ == "__main__":
    for fuel in ["Pb95", "Pb98", "ON", "LPG"]:
        print(predict_fuel_price(fuel))
