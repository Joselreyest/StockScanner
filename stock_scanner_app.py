

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

# Helper function for debug logging

# ======= Debug logging helper =======
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

# ======= Email Alert Function =======
def send_email_alert(results_df, recipient):
    try:
        if not recipient:
            log_debug("Email alert skipped: No recipient provided.")
            return

        log_debug(f"Preparing to send email alert to {recipient} with {len(results_df)} matching stocks.")

        subject = "Stock Scanner Alert - Matching Stocks Found"
        body = results_df.to_string(index=False)

        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = os.getenv("EMAIL_FROM")
        msg["To"] = recipient

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(os.getenv("EMAIL_FROM"), os.getenv("EMAIL_PASSWORD"))
            server.send_message(msg)

        log_debug("Email alert sent successfully.")
    except Exception as e:
        log_debug(f"Failed to send email alert: {e}")
        
# UI: Sidebar
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")
    st.checkbox("Enable Debug Mode", key="enable_debug")
    debug_enabled = st.session_state.get("enable_debug", False)
    st.text_input("Email to notify (optional)", key="alert_email")

st.title("📈 Stock Strategy Scanner")



# Stock Function
def scan_stock(ticker, settings):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')
        if data is None or data.empty or len(data) < 21:
            log_debug(f"Not enough data for {ticker}")
            return None

        latest = data.iloc[-1]
        close, high, low, open_, vol = latest['Close'], latest['High'], latest['Low'], latest['Open'], latest['Volume']

        # RSI calculation
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_rsi) or latest_rsi > settings['rsi_max']:
            log_debug(f"{ticker} filtered out by RSI: {latest_rsi}")
            return None

        if vol < settings['min_volume']:
            log_debug(f"{ticker} filtered out by volume: {vol}")
            return None

        day20_high = data['High'].rolling(window=20).max()
        if high < day20_high.iloc[-2]:
            log_debug(f"{ticker} filtered out by breakout check")
            return None

        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= settings['volume_spike_factor'] * avg_vol.iloc[-2]:
            log_debug(f"{ticker} filtered out by volume spike check")
            return None

        prev_high = data['High'].iloc[-2]
        if open_ < settings['gap_up_factor'] * prev_high:
            log_debug(f"{ticker} filtered out by gap-up check")
            return None

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": True,
            "Volume Spike": True,
            "Gap Up": True,
        }
    except Exception as e:
        log_debug(f"Error fetching {ticker}: {e}")
        return None

# Load S&P 500 data and metadata
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
symbols_500 = sp500["Symbol"].tolist()
sp500_metadata = sp500.set_index("Symbol")[["GICS Sector", "GICS Sub-Industry"]].to_dict("index")

small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

# Market selection
st.subheader("Select Market Segment")
market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])
symbols, uploaded_file = [], None

if market == "S&P 500":
    symbols = symbols_500
    sectors = sorted(sp500["GICS Sector"].dropna().unique())
    selected_sectors = st.multiselect("Filter by Sector (Optional)", sectors)
    if selected_sectors:
        symbols = [s for s in symbols if sp500_metadata.get(s, {}).get("GICS Sector") in selected_sectors]
elif market == "Small Caps":
    symbols = small_caps
else:
    uploaded_file = st.file_uploader("Upload CSV with one ticker per line")
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file, header=None)
            symbols = user_df[0].dropna().unique().tolist()
        except:
            st.error("⚠️ Error reading the uploaded file. Make sure it's a CSV with one column of tickers.")

# Strategy filters
# Strategy Parameters
with st.expander("🔧 Strategy Filters"):
    min_volume = st.slider("Minimum Volume", 100_000, 10_000_000, 1_000_000, step=100_000)
    rsi_max = st.slider("Max RSI", 10, 90, 30)
    volume_spike_factor = st.slider("Volume Spike Factor", 1.0, 5.0, 2.0, step=0.1)
    gap_up_factor = st.slider("Gap-Up Factor", 1.0, 1.2, 1.02, step=0.01)

settings = {
    "min_volume": min_volume,
    "rsi_max": rsi_max,
    "volume_spike_factor": volume_spike_factor,
    "gap_up_factor": gap_up_factor,
}

# Select stocks to scan
selected = st.multiselect("Select stocks to scan (or leave empty to scan all)", symbols)
tickers = selected if selected else symbols

# Initialize scan results in session state
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# Scan button logic
if st.button("🔍 Scan Now"):
    st.info(f"Scanning {len(tickers)} stocks...")
    results = []
    for sym in tickers:
        res = scan_stock(sym, settings)
        if res:
            results.append(res)
    if results:
        df = pd.DataFrame(results)
        st.session_state.scan_results = df
        st.success(f"Found {len(df)} matches")

        # Trigger email alert async
        email_to = st.session_state.get("alert_email")
        if email_to:
            threading.Thread(target=send_email_alert, args=(df, email_to), daemon=True).start()
    else:
        st.session_state.scan_results = None
        st.warning("No stocks matched the criteria.")

# Display results and chart from session state if available
if st.session_state.scan_results is not None:
    df = st.session_state.scan_results
    st.dataframe(df)
    st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")

    symbol_select = st.selectbox("Select a symbol to view chart", df["Ticker"].tolist())
    if symbol_select:
        chart_data = yf.Ticker(symbol_select).history(period="1mo")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
                                     low=chart_data['Low'], close=chart_data['Close'], name="Candlestick"))
        st.plotly_chart(fig, use_container_width=True)

# Show debug logs
debug_enabled = st.session_state.get("enable_debug", False)
if debug_enabled and "debug_log" in st.session_state:
    with st.expander("🧞 Debug Log"):
        for entry in st.session_state.debug_log:
            st.text(entry)
