# ⛽ Predyktor Cen Paliw ORLEN

Projekt ML do predykcji hurtowych cen paliw PKN ORLEN na 14 dni do przodu.

## Struktura projektu

```
fuel-price-prediction/
├── main.py                        # Pipeline: dane → trening → ewaluacja
├── requirements.txt
├── data/
│   ├── generate_sample_data.py    # Generator przykładowych danych
│   ├── prepare_data.py            # Preprocessing, cechy lag/rolling
│   ├── raw/                       # Surowe dane CSV
│   └── processed/                 # Przetworzone dane CSV
├── model/
│   ├── train_model.py             # RandomForestRegressor + TimeSeriesSplit
│   ├── evaluate.py                # MAE, RMSE, R² per rodzaj paliwa
│   └── predict.py                 # Predykcja dla wybranego paliwa
├── app/
│   └── streamlit_app.py           # UI webowe z wykresem Plotly
├── models/                        # Zapisany model .pkl
└── tests/
    └── test_basic.py              # Testy jednostkowe
```

## Uruchomienie

```bash
# 1. Zainstaluj zależności
pip install -r requirements.txt

# 2. Uruchom pełny pipeline (dane + trening + ewaluacja)
python main.py

# 3. Uruchom aplikację webową
streamlit run app/streamlit_app.py

# 4. Testy jednostkowe
python -m pytest tests/
```

## Model

- **Algorytm:** RandomForestRegressor (scikit-learn)
- **Walidacja:** TimeSeriesSplit (5 foldów) – brak wycieku danych z przyszłości
- **Horyzont predykcji:** 14 dni
- **Cechy:** cena ropy Brent, kurs USD/PLN, marża paliwowa, lag 1/7/14 dni, rolling mean 7/14/30 dni, cechy czasowe

## Rodzaje paliw

| Symbol | Opis           |
|--------|----------------|
| Pb95   | Benzyna 95     |
| Pb98   | Benzyna 98     |
| ON     | Olej napędowy  |
| LPG    | Gaz płynny     |
