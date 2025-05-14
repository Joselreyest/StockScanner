# stock_scanner_app_phase3.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go 
from ta.momentum import RSIIndicator
import datetime
import json

# technical analysis indicators

# Set page config
# ✅ This must come FIRST before any Streamlit UI commands
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

# Sidebar
with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")
    debug_mode = st.checkbox("✅ Enable Debug Mode")
    email_alerts_enabled = st.checkbox("📬 Enable Email Alerts")
    scheduled_run_enabled = st.checkbox("⏰ Enable Scheduled Scan")
    
st.title("📈 Stock Strategy Scanner")

# RSI threshold slider
max_rsi = st.slider("Max RSI", 10, 90, 30)
min_volume = st.slider("Min Volume", 100000, 5000000, 1000000, step=50000)
volume_spike_ratio = st.slider("Volume Spike Ratio (x avg)", 1.0, 5.0, 2.0)
gap_up_pct = st.slider("Gap-up % over prev high", 1, 10, 2)

# Load tickers
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
sp500 = sp500.rename(columns={"Symbol": "Ticker"})
symbols_500 = sp500["Ticker"].tolist()
small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])
symbols = []
uploaded_file = None

if market == "S&P 500":
    sectors = sp500["GICS Sector"].unique().tolist()
    selected_sectors = st.multiselect("Filter by Sector", sectors, default=sectors)

    industries = sp500[sp500["GICS Sector"].isin(selected_sectors)]["GICS Sub-Industry"].unique().tolist()
    selected_industries = st.multiselect("Filter by Industry", industries, default=industries)

    filtered_df = sp500[(sp500["GICS Sector"].isin(selected_sectors)) & (sp500["GICS Sub-Industry"].isin(selected_industries))]
    symbols = filtered_df["Ticker"].tolist()
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
            

# Scanning function
def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            if debug_mode:
                st.write(f"{ticker}: ❌ Not enough data")
            return None, None

        latest = data.iloc[-1]
        close = latest['Close']
        high = latest['High']
        low = latest['Low']
        open_ = latest['Open']
        vol = latest['Volume']

    # RSI
        data['RSI'] = RSIIndicator(data['Close']).rsi()
        latest_rsi = data['RSI'].iloc[-1]

        if pd.isna(latest_rsi) or latest_rsi > max_rsi:
            if debug_mode:
                st.write(f"{ticker}: ❌ RSI {latest_rsi:.2f} exceeds threshold")
            return None, data

        if vol < min_volume:
            if debug_mode:
                st.write(f"{ticker}: ❌ Volume {vol} below threshold")
            return None, data
    
    # Breakout logic
        day20_high = data['High'].rolling(window=20).max()
        if high < day20_high.iloc[-2]:
            if debug_mode:
                st.write(f"{ticker}: ❌ Not a breakout")
            return None, data

    # Volume spike logic        
        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= volume_spike_ratio * avg_vol.iloc[-2]:
            if debug_mode:
                st.write(f"{ticker}: ❌ No volume spike")
            return None, data

        prev_high = data['High'].iloc[-2]
        if open_ < (1 + gap_up_pct / 100) * prev_high:
            if debug_mode:
                st.write(f"{ticker}: ❌ No gap-up")
            return None, data

        score = 0
        if latest_rsi <= max_rsi:
            score += 1
        if vol > min_volume:
            score += 1
        if high > day20_high.iloc[-2]:
            score += 1
        if vol > volume_spike_ratio * avg_vol.iloc[-2]:
            score += 1
        if open_ > (1 + gap_up_pct / 100) * prev_high:
            score += 1
            
        
        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Score": score            
        }, data

    except Exception as e:
        if debug_mode:
            st.write(f"{ticker}: ⚠️ Exception occurred - {e}")
        return None, None

# Load preset
preset_file = "presets.json"
if st.button("💾 Load Last Preset"):
    try:
        with open(preset_file) as f:
            preset = json.load(f)
            max_rsi = preset['max_rsi']
            min_volume = preset['min_volume']
            volume_spike_ratio = preset['volume_spike_ratio']
            gap_up_pct = preset['gap_up_pct']
            st.success("Preset loaded successfully")
    except:
        st.error("No preset found")
        
# Main app execution
# UI for stock selection

if symbols:
    selected = st.multiselect("Select stocks to scan (or leave empty to scan all)", symbols)
    btn = st.button("🔍 Scan Now")

    if btn:
        tickers = selected if selected else symbols
        st.info(f"Scanning {len(tickers)} stocks...")

        results = []
        chart_data = {}
        for sym in tickers:
            res, hist_data = scan_stock(sym)
            if res:
                results.append(res)
                chart_data[sym] = hist_data

        if results:
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.success(f"Found {len(df)} matches")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")

            st.subheader("📊 Charts")
            selected_chart = st.selectbox("Select a stock to view chart", list(chart_data.keys()))

            if selected_chart:
                data = chart_data[selected_chart]

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'))

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    yaxis='y2',
                    line=dict(color='blue')))

                fig.update_layout(
                    title=f"{selected_chart} - Candlestick & RSI",
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(title='Price'),
                    yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

            if email_alerts_enabled:
                st.info("📧 Email alerts are enabled. Integration pending.")
            if scheduled_run_enabled:
                st.info("⏰ Scheduled scanning is enabled. Integration pending.")

            save_btn = st.button("💾 Save This Preset")
            if save_btn:
                with open(preset_file, "w") as f:
                    json.dump({
                        "max_rsi": max_rsi,
                        "min_volume": min_volume,
                        "volume_spike_ratio": volume_spike_ratio,
                        "gap_up_pct": gap_up_pct
                    }, f)
                st.success("Preset saved successfully!")
        else:
            st.warning("No stocks matched the criteria.")

