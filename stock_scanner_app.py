import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime, time as dt_time
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

PRESETS_DIR = "presets"
os.makedirs(PRESETS_DIR, exist_ok=True)

@st.cache_data

def load_sp500_metadata():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

sp500_df = load_sp500_metadata()

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")

    st.subheader("Scanner Settings")
    rsi_threshold = st.slider("Max RSI", 0, 100, 30)
    volume_threshold = st.number_input("Min Volume", value=1_000_000, step=100_000)
    gap_percent = st.slider("Min Gap-up %", 0, 10, 2)
    volume_spike_factor = st.slider("Volume Spike Multiplier", 1, 10, 2)

    st.subheader("Advanced")
    enable_debug = st.checkbox("✅ Debug Mode")
    score_mode = st.checkbox("Enable Scoring Mode")

    if score_mode:
        st.subheader("⚖️ Scoring Weights")
        weight_rsi = st.slider("Weight: RSI Match", 0, 5, 1)
        weight_volume = st.slider("Weight: Volume Match", 0, 5, 1)
        weight_breakout = st.slider("Weight: Breakout Match", 0, 5, 1)
        weight_spike = st.slider("Weight: Volume Spike", 0, 5, 1)
        weight_gap = st.slider("Weight: Gap-up Match", 0, 5, 1)
        min_score = st.slider("Minimum Score to Show", 0, 25, 3)
    else:
        weight_rsi = weight_volume = weight_breakout = weight_spike = weight_gap = 1
        min_score = 0

    st.subheader("📊 Sector/Industry Filter")
    selected_sector = st.selectbox("Filter by Sector (S&P 500 only)", ["All"] + sorted(sp500_df['GICS Sector'].unique()))
    selected_industry = st.selectbox("Filter by Industry (S&P 500 only)", ["All"] + sorted(sp500_df['GICS Sub-Industry'].unique()))

    st.subheader("💾 Presets")
    preset_name = st.text_input("Preset Name")
    if st.button("Save Preset") and preset_name:
        preset = {
            "rsi": rsi_threshold,
            "volume": volume_threshold,
            "gap": gap_percent,
            "spike": volume_spike_factor
        }
        with open(os.path.join(PRESETS_DIR, f"{preset_name}.json"), "w") as f:
            json.dump(preset, f)
        st.success(f"Preset '{preset_name}' saved.")

    preset_files = [f.replace(".json", "") for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
    selected_preset = st.selectbox("Load Preset", ["--Select--"] + preset_files)
    if selected_preset != "--Select--":
        with open(os.path.join(PRESETS_DIR, f"{selected_preset}.json"), "r") as f:
            preset = json.load(f)
            rsi_threshold = preset.get("rsi", rsi_threshold)
            volume_threshold = preset.get("volume", volume_threshold)
            gap_percent = preset.get("gap", gap_percent)
            volume_spike_factor = preset.get("spike", volume_spike_factor)
        st.success(f"Preset '{selected_preset}' loaded.")

    st.subheader("📧 Email Alerts")
    enable_email = st.checkbox("Enable Email Alert")
    user_email = st.text_input("Your Gmail", placeholder="you@gmail.com")
    app_password = st.text_input("App Password", type="password")

    st.subheader("🕒 Scheduled Scan")
    enable_schedule = st.checkbox("Enable Scheduled Scanning")
    scheduled_hour = st.slider("Hour (24h)", 0, 23, 9)
    scheduled_minute = st.slider("Minute", 0, 59, 0)

st.title("📈 Stock Strategy Scanner")

def send_email_alert(matches, user_email, app_password):
    msg = EmailMessage()
    msg['Subject'] = "📈 Stock Strategy Alert - Matches Found"
    msg['From'] = user_email
    msg['To'] = user_email

    body = "The following stocks matched your strategy:\n\n"
    body += matches.to_string(index=False)
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(user_email, app_password)
        smtp.send_message(msg)

def log_debug(message):
    if enable_debug:
        st.session_state.debug_log.append(message)
        print(message)

def scan_stock(ticker, selected_sector, selected_industry):
    try:
        log_debug(f"\n--- Scanning {ticker} ---")

        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        if selected_sector != "All" and sector != selected_sector:
            log_debug(f"Filtered by sector: {sector}")
            return None
        if selected_industry != "All" and industry != selected_industry:
            log_debug(f"Filtered by industry: {industry}")
            return None

        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            log_debug(f"Not enough data for {ticker}")
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

        log_debug(f"RSI for {ticker}: {latest_rsi:.2f}")

        score = 0

        if not pd.isna(latest_rsi) and latest_rsi <= rsi_threshold:
            score += weight_rsi
        else:
            log_debug(f"RSI too high for {ticker}")
            if not score_mode:
                return None

        if vol >= volume_threshold:
            score += weight_volume
        else:
            log_debug(f"Volume too low for {ticker}: {vol}")
            if not score_mode:
                return None

        day20_high = data['High'].rolling(window=20).max()
        if high >= day20_high.iloc[-2]:
            score += weight_breakout
        else:
            log_debug(f"No breakout for {ticker}")
            if not score_mode:
                return None

        avg_vol = data['Volume'].rolling(20).mean()
        if not pd.isna(avg_vol.iloc[-2]) and vol > volume_spike_factor * avg_vol.iloc[-2]:
            score += weight_spike
        else:
            log_debug(f"No volume spike for {ticker}")
            if not score_mode:
                return None

        prev_high = data['High'].iloc[-2]
        if open_ >= (1 + gap_percent / 100) * prev_high:
            score += weight_gap
        else:
            log_debug(f"No gap up for {ticker}")
            if not score_mode:
                return None

        log_debug(f"✅ {ticker} matched with score {score}/25")

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Score": score,
            "Sector": sector,
            "Chart": data[-30:].copy()
        }

    except Exception as e:
        log_debug(f"Error fetching {ticker}: {e}")
        st.error(f"Error fetching {ticker}: {e}")
        return None

def run_scan(tickers):
    st.session_state.debug_log = []
    results = []
    charts = {}
    for sym in tickers:
        res = scan_stock(sym, selected_sector, selected_industry)
        if res:
            if not score_mode or res["Score"] >= min_score:
                charts[sym] = res.pop("Chart")
                results.append(res)
    st.session_state.scan_results = results
    st.session_state.scan_charts = charts

# ... rest of file remains unchanged ...
