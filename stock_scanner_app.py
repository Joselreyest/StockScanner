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

# UI: Sidebar
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")
    st.checkbox("Enable Debug Mode", key="enable_debug")
    debug_enabled = st.session_state.get("enable_debug", False)
    st.text_input("Email to notify (optional)", key="alert_email")
    st.text_input("Exclude tickers (comma separated)", key="exclude_tickers")
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

def send_email_alert(results_df, recipient):
    try:
        if not recipient:
            return

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

def scan_stock(ticker, settings):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')
        if data is None or data.empty or len(data) < 21:
            log_debug(f"Not enough data for {ticker}")
            return None

        latest = data.iloc[-1]
        close, high, low, open_, vol = latest['Close'], latest['High'], latest['Low'], latest['Open'], latest['Volume']

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

        score = 0
        score += (30 - latest_rsi) * 0.2
        score += (vol / avg_vol.iloc[-2]) * 2

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": True,
            "Volume Spike": True,
            "Gap Up": True,
            "Score": round(score, 2)
        }
    except Exception as e:
        log_debug(f"Error fetching {ticker}: {e}")
        return None

# [Rest of the unchanged code continues...]
