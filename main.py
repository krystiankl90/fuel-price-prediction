"""
Główny skrypt pipeline'u: Predykcja Cen Paliw ORLEN.

Uruchamia kolejno:
1. Generowanie przykładowych danych CSV (jeśli brak pliku),
2. Przygotowanie i preprocessing danych,
3. Trenowanie modelu RandomForest,
4. Ewaluację modelu i wyświetlenie metryk.

Użycie:
    python main.py
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dodaj katalog projektu do sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DATA_PATH       = os.path.join("data", "raw", "fuel_prices.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "fuel_prices_processed.csv")


def step_generate_sample_data() -> None:
    if os.path.exists(RAW_DATA_PATH):
        logger.info("Plik z danymi już istnieje: %s (pomijam generowanie)", RAW_DATA_PATH)
        return
    logger.info("=== KROK 1: Generowanie przykładowych danych ===")
    from data.generate_sample_data import generate_sample_csv
    generate_sample_csv(RAW_DATA_PATH)
    logger.info("Przykładowe dane wygenerowane.")


def step_prepare_data() -> None:
    logger.info("=== KROK 2: Przygotowanie danych ===")
    from data.prepare_data import prepare_data
    df = prepare_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)
    logger.info("Przygotowano %d wierszy danych.", len(df))


def step_train_model() -> None:
    logger.info("=== KROK 3: Trenowanie modelu ===")
    from model.train_model import run_training_pipeline
    metrics = run_training_pipeline()
    logger.info("Model wytrenowany. Metryki: %s", metrics)


def step_evaluate_model() -> None:
    logger.info("=== KROK 4: Ewaluacja modelu ===")
    from model.evaluate import run_evaluation
    results = run_evaluation()
    logger.info("Ewaluacja zakończona. Wyniki: %s", results)


def main() -> None:
    logger.info("### START – Pipeline predykcji cen paliw ORLEN ###")
    step_generate_sample_data()
    step_prepare_data()
    step_train_model()
    step_evaluate_model()
    logger.info("### KONIEC – Pipeline zakończony pomyślnie ###")
    logger.info("Uruchom aplikację webową: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
