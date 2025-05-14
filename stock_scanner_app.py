import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def run_scan_button():
    if st.button("🔍 Run Scan"):
        tickers = sp500_df['Symbol'].tolist()
        run_scan(tickers)

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

    if enable_schedule:
        def schedule_loop():
            while True:
                now = datetime.now()
                if now.hour == scheduled_hour and now.minute == scheduled_minute:
                    tickers = sp500_df['Symbol'].tolist()
                    run_scan(tickers)
                    time.sleep(60)
                time.sleep(5)

        threading.Thread(target=schedule_loop, daemon=True).start()

st.title("📈 Stock Strategy Scanner")
run_scan_button()

# Remaining code (send_email_alert, log_debug, scan_stock, run_scan, display_results, etc.) continues unchanged
