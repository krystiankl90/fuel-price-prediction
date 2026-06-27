#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f "models/fuel_price_model.pkl" ]; then
    echo "Trening modelu (pierwsze uruchomienie)..."
    python3 main.py
fi

sleep 1 && open http://localhost:8501 &
streamlit run app/streamlit_app.py --client.toolbarMode minimal
