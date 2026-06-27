"""
Moduł generowania przykładowych danych CSV dla projektu predykcji cen paliw.
"""

import os
import numpy as np
import pandas as pd

FUEL_TYPES = ["Pb95", "Pb98", "ON", "LPG"]

BASE_PRICES = {
    "Pb95": 5.80,
    "Pb98": 6.30,
    "ON":   6.00,
    "LPG":  2.90,
}

BASE_BRENT_USD = 75.0
BASE_USD_PLN   = 4.05
BASE_MARGIN    = 0.40


def _generate_price_series(dates, base_price, volatility=0.015, trend=0.0002, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dates)
    shocks = rng.normal(0, volatility, n)
    month = dates.month.values
    seasonal = 0.12 * np.sin(2 * np.pi * (month - 3) / 12)
    prices = base_price + np.cumsum(shocks) + trend * np.arange(n) + seasonal
    prices = np.clip(prices, base_price * 0.6, base_price * 1.6)
    return np.round(prices, 3)


def generate_sample_csv(output_path: str) -> pd.DataFrame:
    """
    Generuje przykładowy plik CSV z danymi historycznymi cen paliw.
    Dane tygodniowe dla lat 2018-2019 i 2022-2025 (pomija COVID 2020-2021).
    """
    dates_pre_covid  = pd.date_range("2018-01-01", "2019-12-31", freq="W")
    dates_post_covid = pd.date_range("2022-01-01", "2025-12-31", freq="W")
    all_dates = dates_pre_covid.append(dates_post_covid)

    n = len(all_dates)
    rng = np.random.default_rng(0)

    brent_prices  = np.clip(BASE_BRENT_USD + np.cumsum(rng.normal(0, 1.5, n)), 40, 130).round(2)
    usd_pln_rates = np.clip(BASE_USD_PLN   + np.cumsum(rng.normal(0, 0.005, n)), 3.50, 5.00).round(4)
    fuel_margins  = np.clip(BASE_MARGIN    + rng.normal(0, 0.03, n), 0.10, 0.80).round(3)

    rows = []
    for i, fuel_type in enumerate(FUEL_TYPES):
        prices = _generate_price_series(all_dates, BASE_PRICES[fuel_type], seed=42 + i * 7)
        for j, date in enumerate(all_dates):
            rows.append({
                "date":            date.date(),
                "fuel_type":       fuel_type,
                "fuel_price_pln":  prices[j],
                "brent_price_usd": brent_prices[j],
                "usd_pln_rate":    usd_pln_rates[j],
                "fuel_margin_pln": fuel_margins[j],
            })

    dataframe = pd.DataFrame(rows).sort_values(["date", "fuel_type"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"[INFO] Wygenerowano {len(dataframe)} wierszy → {output_path}")
    return dataframe


if __name__ == "__main__":
    generate_sample_csv(os.path.join("data", "raw", "fuel_prices.csv"))
