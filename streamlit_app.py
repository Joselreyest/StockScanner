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
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

@st.cache_data
def get_nasdaq_symbols():
    url = "https://api.nasdaq.com/api/screener/stocks?exchange=nasdaq&download=true"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()
    tickers = [row["symbol"] for row in data["data"]["rows"]]
    return tickers

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

def send_email_alert(subject, body=None, html=None):
    receiver = st.session_state.get("alert_email")
    if not receiver:
        return

    msg = EmailMessage()
    if html:
        msg.set_content("This email contains an HTML table. Please view in an HTML-compatible email client.")
        msg.add_alternative(html, subtype='html')
    else:
        msg.set_content(body or "No data available.")

    msg["Subject"] = subject
    msg["From"] = "noreply@stockscanner.app"
    msg["To"] = receiver

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
            server.send_message(msg)
    except Exception as e:
        log_debug(f"Failed to send email: {e}")
        
def scan_stock(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1mo")
        if df.empty or len(df) < 15:
            log_debug(f"Failed: Not enough data for {symbol}")
            return None
        df["RSI"] = compute_rsi(df["Close"])
        if df["RSI"].isnull().all():
            log_debug(f"Failed: RSI is all None for {symbol}")
            return None
        df["Volume Spike"] = df["Volume"] > (df["Volume"].shift(1) * st.session_state.volume_spike_factor)
        gap_up = df["Open"].iloc[-1] > df["Close"].iloc[-2] * st.session_state.gap_up_factor
        rsi_cond = df["RSI"].iloc[-1] < st.session_state.rsi_max
        volume_cond = df["Volume"].iloc[-1] > st.session_state.min_volume

        failed_criteria = []
        if not gap_up: failed_criteria.append("Gap Up")
        if not rsi_cond: failed_criteria.append("RSI")
        if not volume_cond: failed_criteria.append("Volume")

        if failed_criteria:
            log_debug(f"{symbol} failed: {', '.join(failed_criteria)}")
            return {
                "Symbol": symbol,
                "Price": df["Close"].iloc[-1],
                "RSI": df["RSI"].iloc[-1],
                "Volume": df["Volume"].iloc[-1],
                "Gap Up": gap_up,
                "Score": 0,
                "Reason": ", ".join(failed_criteria)
            }
        else:
            score = sum([gap_up, rsi_cond, df["Volume Spike"].iloc[-1]])
            return {
                "Symbol": symbol,
                "Price": df["Close"].iloc[-1],
                "RSI": df["RSI"].iloc[-1],
                "Volume": df["Volume"].iloc[-1],
                "Gap Up": gap_up,
                "Score": score,
                "Reason": "Matched"
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

def format_email_table(df):
    try:
        styled = df[["Symbol", "Price", "RSI", "Volume", "Gap Up", "Score", "Reason"]].copy()
        html_table = styled.to_html(index=False, border=0)
        style_block = """
        <style>
            table { border-collapse: collapse; width: 100%; font-family: Arial; }
            th, td { border: 1px solid #dddddd; text-align: center; padding: 8px; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            th { background-color: #4CAF50; color: white; }
        </style>
        """
        return style_block + html_table
    except Exception as e:
        log_debug(f"Error formatting email table: {e}")
        return "<p>Failed to render table</p>"
        
def plot_chart(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo")
        data["RSI"] = compute_rsi(data["Close"])
        data["SMA20"] = data["Close"].rolling(window=20).mean()
        data["SMA50"] = data["Close"].rolling(window=50).mean()
        data["UpperBB"] = data["Close"].rolling(window=20).mean() + 2*data["Close"].rolling(window=20).std()
        data["LowerBB"] = data["Close"].rolling(window=20).mean() - 2*data["Close"].rolling(window=20).std()

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["UpperBB"], mode="lines", name="Upper BB", line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["LowerBB"], mode="lines", name="Lower BB", line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=2, col=1)

        macd_line = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode="lines", name="MACD Line"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode="lines", name="Signal Line"), row=3, col=1)

        fig.update_layout(title=f"{symbol} Chart with Indicators")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load chart: {e}")

def perform_daily_scan():
    results = []
    excluded = [x.strip().upper() for x in st.session_state.get("exclude_tickers", "").split(",") if x]
    log_debug(f"Starting scan with {len(ticker_list)} tickers.")
    log_debug(f"Excluded tickers: {excluded}")
    progress_bar = st.progress(0, text="Scanning tickers...")

    for i, ticker in enumerate(ticker_list):
        if ticker in excluded:
            log_debug(f"Skipping excluded ticker: {ticker}")
            continue

        log_debug(f"Scanning {ticker}")
        res = scan_stock(ticker)

        if res:
            log_debug(f"Match found: {res}")
            results.append(res)
        else:
            log_debug(f"No match for {ticker}")

        progress_bar.progress((i + 1) / len(ticker_list))

    progress_bar.empty()

    if results:
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        def highlight_row(row):
            color = "#d4edda" if row.get("Reason") == "Matched" else "#f8d7da"
            return ["background-color: {}".format(color)] * len(row)

        st.dataframe(df.style.apply(highlight_row, axis=1))

        if "alert_email" in st.session_state and st.session_state.alert_email:
            body = df.to_string(index=False)

        if "results_df" in locals() and not results_df.empty:
            html_body = format_email_table(results_df)
            send_email_alert("Stock Scanner Alert", html=html_body)
        else:
            send_email_alert("Stock Scanner Alert", body="No matches found for today's scan.")
    
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

    source_option = st.selectbox("Select Ticker Source", ["NASDAQ", "S&P 500", "Small Caps", "Upload CSV"])    
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

if st.session_state.get("enable_debug", False):
    if "debug_log" in st.session_state and st.session_state.debug_log:
        st.subheader("🛠 Debug Log")
        for entry in st.session_state.debug_log:
            st.text(entry)
    else:
        st.info("Debug mode is enabled, but no logs have been generated yet.")
