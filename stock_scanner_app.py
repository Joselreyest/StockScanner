# stock_scanner_app_phase6.py

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime
import json
import os

# technical analysis indicators

# Set page config
# ✅ This must come FIRST before any Streamlit UI commands
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

# Sidebar
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
    if st.button("💾 Save Preset"):
        preset = {
            "rsi": rsi_threshold,
            "volume": volume_threshold,
            "gap": gap_percent,
            "spike": volume_spike_factor
        }
        with open("preset.json", "w") as f:
            json.dump(preset, f)
        st.success("Preset saved.")
    if st.button("📂 Load Preset") and os.path.exists("preset.json"):
        with open("preset.json", "r") as f:
            preset = json.load(f)
            rsi_threshold = preset.get("rsi", rsi_threshold)
            volume_threshold = preset.get("volume", volume_threshold)
            gap_percent = preset.get("gap", gap_percent)
            volume_spike_factor = preset.get("spike", volume_spike_factor)
        st.success("Preset loaded.")

    st.subheader("📧 Email Alerts")
    enable_email = st.checkbox("Enable Email Alert")
    user_email = st.text_input("Your Gmail", placeholder="you@gmail.com")
    app_password = st.text_input("App Password", type="password")
    
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

# Scanning function
def scan_stock(ticker):
    try:
        if enable_debug:
            print(f"\n--- Scanning {ticker} ---")
        
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            if enable_debug:
                print(f"Not enough data for {ticker}")            
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

        if enable_debug:
            print(f"RSI for {ticker}: {latest_rsi:.2f}")
            
        if pd.isna(latest_rsi) or latest_rsi > rsi_threshold:
            if enable_debug:
                print(f"RSI too high for {ticker}")            
            return None

        if vol < volume_threshold:
            if enable_debug:
                print(f"Volume too low for {ticker}: {vol}")            
            return None

        day20_high = data['High'].rolling(window=20).max()
        if high < day20_high.iloc[-2]:
            if enable_debug:
                print(f"No breakout for {ticker}")            
            return None

        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= volume_spike_factor * avg_vol.iloc[-2]:
            if enable_debug:
                print(f"No volume spike for {ticker}")            
            return None

        prev_high = data['High'].iloc[-2]
        if open_ < (1 + gap_percent / 100) * prev_high:
            if enable_debug:
                print(f"No gap up for {ticker}")            
            return None

        if enable_debug:
            print(f"✅ {ticker} matched all criteria")
            
        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": True,
            "Volume Spike": True,
            "Gap Up": True,
            "Chart": data[-30:].copy()
        }

    except Exception as e:
        if enable_debug:
            print(f"Error fetching {ticker}: {e}")            
            st.error(f"Error fetching {ticker}: {e}")
        return None
        
# Load tickers
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
sp500["Sector"] = sp500["GICS Sector"]
sp500["Industry"] = sp500["GICS Sub-Industry"]
symbols_500 = sp500["Symbol"].tolist()

small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

st.subheader("Select Market Segment")
market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])

symbols = []
uploaded_file = None
if market == "S&P 500":
    sectors = st.multiselect("Filter by Sector", sp500["Sector"].unique())
    industries = st.multiselect("Filter by Industry", sp500["Industry"].unique())
    filtered = sp500
    if sectors:
        filtered = filtered[filtered["Sector"].isin(sectors)]
    if industries:
        filtered = filtered[filtered["Industry"].isin(industries)]
    symbols = filtered["Symbol"].tolist()
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

if symbols:
    selected = st.multiselect("Select stocks to scan (or leave empty to scan all)", symbols)
    btn = st.button("🔍 Scan Now")

    if btn:
        tickers = selected if selected else symbols
        st.info(f"Scanning {len(tickers)} stocks...")

        results = []
        charts = {}
        for sym in tickers:
            res = scan_stock(sym)
            if res:
                charts[sym] = res.pop("Chart")
                results.append(res)

        if results:
            df = pd.DataFrame(results)
            st.success(f"Found {len(df)} matches")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")

            chart_ticker = st.selectbox("📊 View Chart for", df["Ticker"].tolist())
            if chart_ticker in charts:
                chart_data = charts[chart_ticker]
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close']
                    )
                ])
                fig.update_layout(title=f"Candlestick Chart: {chart_ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)

            if enable_email and user_email and app_password:
                send_email_alert(df, user_email, app_password)
                st.success("✅ Email alert sent!")
        else:
            st.warning("No stocks matched the criteria.")
