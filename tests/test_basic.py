"""
Testy jednostkowe dla projektu predykcji cen paliw.
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.prepare_data import remove_covid_period, clean_data, add_time_features, add_lag_features
from model.predict import determine_price_direction


class TestRemoveCovidPeriod(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "date": pd.to_datetime(["2019-06-01", "2020-03-15", "2021-07-20", "2022-01-10"]),
            "fuel_price_pln": [5.0, 4.5, 4.8, 6.2],
            "fuel_type": ["Pb95"] * 4,
        })

    def test_removes_covid_years(self):
        result = remove_covid_period(self.df)
        years  = result["date"].dt.year.unique().tolist()
        self.assertNotIn(2020, years)
        self.assertNotIn(2021, years)

    def test_keeps_non_covid_years(self):
        self.assertEqual(len(remove_covid_period(self.df)), 2)


class TestCleanData(unittest.TestCase):
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-08"]),
            "fuel_type": ["Pb95", "Pb95", "Pb95"],
            "fuel_price_pln": [6.0, 6.0, 6.1],
        })
        self.assertEqual(len(clean_data(df)), 2)


class TestAddTimeFeatures(unittest.TestCase):
    def test_adds_month_column(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-03-15"]),
            "fuel_type": ["Pb95"],
            "fuel_price_pln": [6.5],
        })
        result = add_time_features(df)
        self.assertIn("month", result.columns)
        self.assertEqual(result["month"].iloc[0], 3)


class TestDeterminePriceDirection(unittest.TestCase):
    def test_wzrost(self):
        self.assertEqual(determine_price_direction(5.0, 5.2), "wzrost")

    def test_spadek(self):
        self.assertEqual(determine_price_direction(5.5, 5.3), "spadek")

    def test_stabilnie(self):
        self.assertEqual(determine_price_direction(5.0, 5.02), "stabilnie")


if __name__ == "__main__":
    unittest.main()
