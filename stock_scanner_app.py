import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime, timedelta
import json
import os
import threading
import time

st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "scan_charts" not in st.session_state:
    st.session_state.scan_charts = {}
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []
if "last_selected_symbol" not in st.session_state:
    st.session_state.last_selected_symbol = None
if "auto_scan" not in st.session_state:
    st.session_state.auto_scan = False
if "backtest_date" not in st.session_state:
    st.session_state.backtest_date = None

PRESETS_DIR = "presets"
os.makedirs(PRESETS_DIR, exist_ok=True)

@st.cache_data

def load_sp500_metadata():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

sp500_df = load_sp500_metadata()

uploaded_file = st.sidebar.file_uploader("Upload CSV of custom tickers", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if "Symbol" in df_uploaded.columns:
            st.session_state.custom_tickers = df_uploaded["Symbol"].dropna().unique().tolist()
            st.sidebar.success(f"Loaded {len(st.session_state.custom_tickers)} custom tickers.")
        else:
            st.sidebar.error("CSV must contain a 'Symbol' column.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

def log_debug(msg):
    if st.session_state.get("enable_debug"):
        st.session_state.debug_log.append(msg)
        print("DEBUG:", msg)

def send_email_alert(subject, body, to_email, from_email, app_password):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        log_debug("Email sent successfully")
    except Exception as e:
        log_debug(f"Email send failed: {e}")

def scan_stock(ticker, backtest_date=None):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='2mo', interval='1d')
        if data is None or data.empty or len(data) < 21:
            return None

        if backtest_date:
            data = data[data.index <= backtest_date]
            if len(data) < 21:
                return None

        latest = data.iloc[-1]
        close = latest['Close']
        high = latest['High']
        low = latest['Low']
        open_ = latest['Open']
        vol = latest['Volume']

        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        match_rsi = not pd.isna(latest_rsi) and latest_rsi <= st.session_state.rsi_threshold
        match_volume = vol >= st.session_state.volume_threshold
        day20_high = data['High'].rolling(window=20).max()
        match_breakout = high >= day20_high.iloc[-2]
        avg_vol = data['Volume'].rolling(20).mean()
        match_spike = not pd.isna(avg_vol.iloc[-2]) and vol > st.session_state.volume_spike_factor * avg_vol.iloc[-2]
        prev_high = data['High'].iloc[-2]
        match_gap = open_ >= (1 + st.session_state.gap_percent / 100) * prev_high

        if st.session_state.exclude_penny and close < 5:
            return None

        score = (
            st.session_state.weight_rsi * match_rsi +
            st.session_state.weight_volume * match_volume +
            st.session_state.weight_breakout * match_breakout +
            st.session_state.weight_spike * match_spike +
            st.session_state.weight_gap * match_gap
        ) if st.session_state.score_mode else 0

        if st.session_state.score_mode and score < st.session_state.min_score:
            return None

        if not st.session_state.score_mode and not (match_rsi and match_volume and match_breakout and match_spike and match_gap):
            return None

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": match_breakout,
            "Volume Spike": match_spike,
            "Gap Up": match_gap,
            "Score": score
        }
    except Exception as e:
        log_debug(f"Error scanning {ticker}: {e}")
        return None

def run_scan(tickers):
    st.session_state.scan_results = []
    results = []
    for sym in tickers:
        res = scan_stock(sym, backtest_date=st.session_state.backtest_date)
        if res:
            results.append(res)

    st.session_state.scan_results = results
    display_results(results)

    if results and st.session_state.enable_email and st.session_state.user_email and st.session_state.app_password:
        body = f"Stock Scanner found {len(results)} matches:\n\n" + "\n".join([r['Ticker'] for r in results])
        send_email_alert("Stock Scanner Alert", body, st.session_state.user_email, st.session_state.user_email, st.session_state.app_password)

def display_results(results):
    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ Found {len(df)} matching stocks")
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")

        if st.session_state.score_mode:
            st.subheader("Scoring Heatmap")
            st.dataframe(df.set_index("Ticker")[['Score']].style.background_gradient(cmap='YlGnBu'))

        tickers = df['Ticker'].tolist()
        symbol = st.selectbox("📊 Select a stock to view chart", tickers)
        if symbol:
            show_stock_chart(symbol)
            show_future_performance(symbol)
    else:
        st.warning("No matching stocks found.")

def show_stock_chart(ticker):
    data = yf.Ticker(ticker).history(period='1mo', interval='1d')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def show_future_performance(ticker):
    st.subheader("📉 Post-Match Performance (Simulated)")
    data = yf.Ticker(ticker).history(period='1mo', interval='1d')
    match_date = st.session_state.backtest_date or data.index[-1]
    future = data[data.index > match_date].head(5)
    if future.empty:
        st.info("No future data available.")
        return

    future['Pct Change'] = future['Close'].pct_change().fillna(0).cumsum()
    st.line_chart(future['Pct Change'])
    st.dataframe(future[['Close', 'Volume', 'Pct Change']])

with st.sidebar:
    st.session_state.rsi_threshold = st.slider("Max RSI", 0, 100, 30)
    st.session_state.volume_threshold = st.number_input("Min Volume", value=1_000_000, step=100_000)
    st.session_state.gap_percent = st.slider("Min Gap-up %", 0, 10, 2)
    st.session_state.volume_spike_factor = st.slider("Volume Spike Multiplier", 1, 10, 2)
    st.session_state.exclude_penny = st.checkbox("Exclude Penny Stocks (<$5)", value=True)
    st.session_state.enable_debug = st.checkbox("✅ Debug Mode")
    st.session_state.score_mode = st.checkbox("Enable Scoring Mode")
    if st.session_state.score_mode:
        st.session_state.weight_rsi = st.slider("Weight: RSI Match", 0, 5, 1)
        st.session_state.weight_volume = st.slider("Weight: Volume Match", 0, 5, 1)
        st.session_state.weight_breakout = st.slider("Weight: Breakout Match", 0, 5, 1)
        st.session_state.weight_spike = st.slider("Weight: Volume Spike", 0, 5, 1)
        st.session_state.weight_gap = st.slider("Weight: Gap-up Match", 0, 5, 1)
        st.session_state.min_score = st.slider("Minimum Score to Show", 0, 25, 3)
    else:
        st.session_state.weight_rsi = st.session_state.weight_volume = st.session_state.weight_breakout = st.session_state.weight_spike = st.session_state.weight_gap = 1
        st.session_state.min_score = 0

    st.session_state.backtest_date = st.date_input("📅 Backtest Date (Optional)", value=None)

    if st.checkbox("💾 Save Current Preset"):
        name = st.text_input("Preset Name")
        if name:
            preset = {
                "rsi_threshold": st.session_state.rsi_threshold,
                "volume_threshold": st.session_state.volume_threshold,
                "gap_percent": st.session_state.gap_percent,
                "volume_spike_factor": st.session_state.volume_spike_factor,
                "exclude_penny": st.session_state.exclude_penny,
                "score_mode": st.session_state.score_mode,
                "weights": {
                    "rsi": st.session_state.weight_rsi,
                    "volume": st.session_state.weight_volume,
                    "breakout": st.session_state.weight_breakout,
                    "spike": st.session_state.weight_spike,
                    "gap": st.session_state.weight_gap,
                    "min_score": st.session_state.min_score,
                }
            }
            with open(f"{PRESETS_DIR}/{name}.json", "w") as f:
                json.dump(preset, f)
            st.success("Preset saved")

    preset_files = os.listdir(PRESETS_DIR)
    if preset_files:
        selected_preset = st.selectbox("📂 Load Preset", ["None"] + preset_files)
        if selected_preset != "None":
            with open(f"{PRESETS_DIR}/{selected_preset}", "r") as f:
                preset = json.load(f)
            st.session_state.rsi_threshold = preset["rsi_threshold"]
            st.session_state.volume_threshold = preset["volume_threshold"]
            st.session_state.gap_percent = preset["gap_percent"]
            st.session_state.volume_spike_factor = preset["volume_spike_factor"]
            st.session_state.exclude_penny = preset.get("exclude_penny", True)
            st.session_state.score_mode = preset["score_mode"]
            if st.session_state.score_mode:
                st.session_state.weight_rsi = preset["weights"]["rsi"]
                st.session_state.weight_volume = preset["weights"]["volume"]
                st.session_state.weight_breakout = preset["weights"]["breakout"]
                st.session_state.weight_spike = preset["weights"]["spike"]
                st.session_state.weight_gap = preset["weights"]["gap"]
                st.session_state.min_score = preset["weights"]["min_score"]
            st.success("Preset loaded")

    st.session_state.auto_scan = st.checkbox("🔄 Enable Auto Scan (every 15 min)")

st.title("📈 Stock Strategy Scanner")

def run_scan_button():
    if st.button("🔍 Run Scan"):
        if 'custom_tickers' in st.session_state and st.session_state.custom_tickers:
            tickers = st.session_state.custom_tickers
        else:
            tickers = sp500_df['Symbol'].tolist()

        if 'selected_sector' in st.session_state and st.session_state.selected_sector != "All":
            tickers = sp500_df[sp500_df['GICS Sector'] == st.session_state.selected_sector]['Symbol'].tolist()
        if 'selected_industry' in st.session_state and st.session_state.selected_industry != "All":
            tickers = sp500_df[sp500_df['GICS Sub-Industry'] == st.session_state.selected_industry]['Symbol'].tolist()

        run_scan(tickers)

run_scan_button()

def auto_scan_loop():
    while True:
        if st.session_state.auto_scan:
            log_debug("Auto-scan triggered")
            run_scan(sp500_df['Symbol'].tolist())
        time.sleep(900)  # 15 minutes

if st.session_state.auto_scan:
    threading.Thread(target=auto_scan_loop, daemon=True).start()
