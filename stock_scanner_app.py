def log_debug(msg):
    debug_enabled = st.session_state.get("enable_debug", False)
    if debug_enabled:
        if "debug_log" not in st.session_state:
            st.session_state.debug_log = []
        st.session_state.debug_log.append(msg)
        try:
            print("DEBUG:", msg)
        except Exception:
            pass

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib, ssl
from email.message import EmailMessage
import os, json, time, threading
import requests
from bs4 import BeautifulSoup
import schedule

st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

@st.cache_data
def get_nasdaq_symbols():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = pd.read_html(str(table))[0]["Ticker"].tolist()
    return [ticker.replace(".", "-") for ticker in tickers]

@st.cache_data
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = pd.read_html(str(table))[0]["Symbol"].tolist()
    return [ticker.replace(".", "-") for ticker in tickers]

@st.cache_data
def get_small_cap_symbols():
    return ["AVXL", "PLTR", "BB", "MVIS", "NNDM", "HIMS"]

def send_email_alert(subject, body):
    receiver = st.session_state.get("alert_email")
    if not receiver:
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = "noreply@stockscanner.app"
    msg["To"] = receiver
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
        server.send_message(msg)

def scan_stock(symbol):
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df.empty or len(df) < 2:
            return None
        df["RSI"] = compute_rsi(df["Close"])
        df["Volume Spike"] = df["Volume"] > (df["Volume"].shift(1) * st.session_state.volume_spike_factor)
        gap_up = df["Open"].iloc[-1] > df["Close"].iloc[-2] * st.session_state.gap_up_factor
        rsi_cond = df["RSI"].iloc[-1] < st.session_state.rsi_max
        volume_cond = df["Volume"].iloc[-1] > st.session_state.min_volume
        score = sum([gap_up, rsi_cond, df["Volume Spike"].iloc[-1]])
        if all([gap_up, rsi_cond, volume_cond]):
            return {
                "Symbol": symbol,
                "RSI": df["RSI"].iloc[-1],
                "Volume": df["Volume"].iloc[-1],
                "Gap Up": gap_up,
                "Score": score
            }
    except Exception as e:
        log_debug(f"Error scanning {symbol}: {e}")
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_chart(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo")
        data["RSI"] = compute_rsi(data["Close"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=2, col=1)
        fig.update_layout(title=f"{symbol} Chart with RSI")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load chart: {e}")

def perform_daily_scan():
    results = []
    excluded = [x.strip().upper() for x in st.session_state.get("exclude_tickers", "").split(",") if x]
    for ticker in ticker_list:
        if ticker in excluded:
            continue
        res = scan_stock(ticker)
        if res:
            results.append(res)
    if results:
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        st.dataframe(df)
        if "alert_email" in st.session_state and st.session_state.alert_email:
            body = df.to_string(index=False)
            send_email_alert("Stock Scanner Alert", body)
        symbol_select = st.selectbox("Select stock to view chart", df["Symbol"])
        plot_chart(symbol_select)
    else:
        st.info("No matches found based on current filters.")

def scheduler():
    schedule.clear()
    if scan_time_input:
        schedule.every().day.at(scan_time_input.strftime("%H:%M")).do(perform_daily_scan)
    while True:
        schedule.run_pending()
        time.sleep(30)

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")
    st.checkbox("Enable Debug Mode", key="enable_debug")
    st.text_input("Email to notify (optional)", key="alert_email")
    st.text_input("Exclude tickers (comma separated)", key="exclude_tickers")
    scan_time_input = st.time_input("Schedule Daily Scan", key="scan_time")

    source_option = st.selectbox("Select Ticker Source", ["NASDAQ 100", "S&P 500", "Small Caps", "Upload CSV"])
    uploaded_file = st.file_uploader("Upload CSV (Ticker column)", type=["csv"]) if source_option == "Upload CSV" else None

    if source_option == "NASDAQ 100":
        ticker_list = get_nasdaq_symbols()
    elif source_option == "S&P 500":
        ticker_list = get_sp500_symbols()
    elif source_option == "Small Caps":
        ticker_list = get_small_cap_symbols()
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        ticker_list = df.iloc[:, 0].dropna().astype(str).str.upper().tolist()
    else:
        ticker_list = []

    st.number_input("Minimum Volume", min_value=0, value=100000, key="min_volume")
    st.slider("Max RSI", min_value=0, max_value=100, value=70, key="rsi_max")
    st.slider("Volume Spike Factor", min_value=1.0, max_value=5.0, value=1.5, step=0.1, key="volume_spike_factor")
    st.slider("Gap Up Factor", min_value=1.0, max_value=2.0, value=1.02, step=0.01, key="gap_up_factor")

    if st.button("💾 Save Filter Preset"):
        preset = {
            "min_volume": st.session_state.min_volume,
            "rsi_max": st.session_state.rsi_max,
            "volume_spike_factor": st.session_state.volume_spike_factor,
            "gap_up_factor": st.session_state.gap_up_factor
        }
        with open("filter_presets.json", "w") as f:
            json.dump(preset, f)
        st.success("Preset saved!")

    if os.path.exists("filter_presets.json"):
        if st.button("📂 Load Filter Preset"):
            with open("filter_presets.json") as f:
                preset = json.load(f)
                for key in preset:
                    st.session_state[key] = preset[key]
            st.success("Preset loaded!")

st.title("📈 Stock Strategy Scanner")

if st.button("▶️ Run Scan Now"):
    perform_daily_scan()

if "scheduler_thread" not in st.session_state:
    thread = threading.Thread(target=scheduler, daemon=True)
    thread.start()
    st.session_state.scheduler_thread = thread
