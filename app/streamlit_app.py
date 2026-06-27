"""
Aplikacja webowa Streamlit – Predyktor Cen Paliw ORLEN.
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.predict import predict_fuel_price  # noqa: E402

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fuel_prices_processed.csv")
FUEL_TYPES = ["Pb95", "Pb98", "ON", "LPG"]

DIRECTION_COLORS = {"wzrost": "#e74c3c", "spadek": "#27ae60", "stabilnie": "#f39c12"}
DIRECTION_EMOJI  = {"wzrost": "📈",       "spadek": "📉",       "stabilnie": "➡️"}


def load_historical_prices(fuel_type: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
    return df[df["fuel_type"] == fuel_type].sort_values("date").tail(52)


def build_price_chart(history, predicted_price, last_date):
    prediction_date  = pd.Timestamp(last_date) + pd.Timedelta(days=14)
    last_known_price = history["fuel_price_pln"].iloc[-1]
    last_known_date  = history["date"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["fuel_price_pln"],
                             mode="lines", name="Historia cen",
                             line={"color": "#3498db", "width": 2}))
    fig.add_trace(go.Scatter(x=[prediction_date], y=[predicted_price],
                             mode="markers", name="Predykcja (14 dni)",
                             marker={"color": "#e74c3c", "size": 14, "symbol": "star"}))
    fig.add_trace(go.Scatter(x=[last_known_date, prediction_date],
                             y=[last_known_price, predicted_price],
                             mode="lines", name="Trend predykcji",
                             line={"color": "#e74c3c", "width": 1.5, "dash": "dash"}))
    fig.update_layout(title="Historia cen i predykcja", xaxis_title="Data",
                      yaxis_title="Cena (PLN/litr)", hovermode="x unified",
                      template="plotly_white", height=400,
                      legend={"orientation": "h"})
    return fig


def display_prediction_metrics(result):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Aktualna cena", f"{result['current_price']:.3f} PLN/l")
    with col2:
        st.metric("Predykcja za 14 dni", f"{result['predicted_price']:.3f} PLN/l",
                  delta=f"{result['price_difference']:+.3f} PLN/l")
    with col3:
        direction = result["direction"]
        color     = DIRECTION_COLORS.get(direction, "#7f8c8d")
        emoji     = DIRECTION_EMOJI.get(direction, "")
        st.markdown(f"""
            <div style='text-align:center;padding:8px 0'>
                <p style='color:#6b7280;font-size:14px;margin-bottom:4px'>Kierunek zmiany</p>
                <p style='font-size:24px;font-weight:bold;color:{color};margin:0'>{emoji} {direction.capitalize()}</p>
            </div>
        """, unsafe_allow_html=True)


def run_app():
    st.set_page_config(page_title="ORLEN – Predyktor Cen Paliw", page_icon="⛽", layout="wide")
    st.title("⛽ Predyktor Cen Paliw ORLEN")
    st.markdown("Aplikacja przewiduje cenę paliwa za **14 dni** na podstawie danych historycznych, ceny ropy Brent i kursu walut.")
    st.divider()

    with st.sidebar:
        st.header("Ustawienia")
        selected_fuel = st.selectbox("Wybierz rodzaj paliwa:", FUEL_TYPES, index=0)
        st.markdown("---")
        st.markdown("**Dostępne paliwa:**\n- **Pb95** – benzyna 95\n- **Pb98** – benzyna 98\n- **ON** – olej napędowy\n- **LPG** – gaz płynny")
        st.markdown("---")
        st.caption("Projekt predykcji cen paliw | ML 2026")

    if not os.path.exists(PROCESSED_DATA_PATH):
        st.error("Nie znaleziono danych. Uruchom najpierw: `python main.py`")
        st.stop()

    with st.spinner(f"Obliczam predykcję dla {selected_fuel}..."):
        try:
            result = predict_fuel_price(selected_fuel)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except ValueError as e:
            st.error(f"Błąd danych: {e}")
            st.stop()

    st.subheader(f"Wyniki predykcji dla: {selected_fuel}")
    display_prediction_metrics(result)

    st.markdown("**Uzasadnienie predykcji:**")
    st.info(result["justification"])
    st.caption(f"Ostatnie dane z: {result['last_data_date']}")

    st.divider()
    st.subheader("Historia cen i punkt predykcji")
    history = load_historical_prices(selected_fuel)
    st.plotly_chart(build_price_chart(history, result["predicted_price"], result["last_data_date"]),
                    use_container_width=True)

    with st.expander("Ostatnie dane historyczne (10 wpisów)"):
        cols = [c for c in ["date", "fuel_price_pln", "brent_price_usd", "usd_pln_rate"] if c in history.columns]
        st.dataframe(history[cols].tail(10).reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    run_app()
