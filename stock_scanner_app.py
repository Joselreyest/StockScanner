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
def send_email_alert(subject, body, to_email, from_email, email_password):
    try:
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        context = ssl.create_default_context()
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(from_email, email_password)
            server.send_message(msg)
        log_debug(f"Email alert sent to {to_email}")
    except Exception as e:
        log_debug(f"Error sending email: {e}")

# ======= Scan Stock Function =======
def scan_stock(ticker, params):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            log_debug(f"{ticker}: Not enough data")
            return None  # Not enough data to analyze

        latest = data.iloc[-1]
        close = latest['Close']
        high = latest['High']
        low = latest['Low']
        open_ = latest['Open']
        vol = latest['Volume']

        # RSI calculation
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_rsi):
            log_debug(f"{ticker}: RSI is NaN")
            return None
        if params["use_rsi"] and latest_rsi > params["rsi_threshold"]:
            log_debug(f"{ticker}: RSI {latest_rsi} above threshold {params['rsi_threshold']}")
            return None

        if params["use_min_volume"] and vol < params["min_volume"]:
            log_debug(f"{ticker}: Volume {vol} below minimum {params['min_volume']}")
            return None

        # Breakout strategy
        if params["use_breakout"]:
            day20_high = data['High'].rolling(window=20).max()
            if high < day20_high.iloc[-2]:  # not a breakout
                log_debug(f"{ticker}: Not breakout")
                return None

        # Volume spike strategy
        if params["use_volume_spike"]:
            avg_vol = data['Volume'].rolling(20).mean()
            if pd.isna(avg_vol.iloc[-2]) or vol <= params["volume_spike_multiplier"] * avg_vol.iloc[-2]:
                log_debug(f"{ticker}: No volume spike")
                return None

        # Gap-up strategy
        if params["use_gap_up"]:
            prev_high = data['High'].iloc[-2]
            if open_ < params["gap_up_min_ratio"] * prev_high:
                log_debug(f"{ticker}: No gap up")
                return None

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": params["use_breakout"],
            "Volume Spike": params["use_volume_spike"],
            "Gap Up": params["use_gap_up"],
        }

    except Exception as e:
        log_debug(f"Error fetching {ticker}: {e}")
        return None

# ======= Main App =======
def main():
    st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

    with st.sidebar:
        st.image("logo.png", width=180)
        st.markdown("**Stock Strategy Scanner**")
        st.caption("by Jose Reyes")

        enable_debug = st.checkbox("Enable Debug Mode")
        st.session_state["enable_debug"] = enable_debug

        st.markdown("---")
        st.subheader("Scan Filters")

        use_rsi = st.checkbox("Filter by RSI", value=True)
        rsi_threshold = st.slider("RSI Threshold (max)", 0, 100, 30)

        use_min_volume = st.checkbox("Filter by Minimum Volume", value=True)
        min_volume = st.number_input("Minimum Volume", min_value=0, value=1_000_000, step=100_000)

        use_breakout = st.checkbox("Use Breakout Strategy", value=True)
        use_volume_spike = st.checkbox("Use Volume Spike Strategy", value=True)
        volume_spike_multiplier = st.slider("Volume Spike Multiplier", 1.0, 5.0, 2.0)

        use_gap_up = st.checkbox("Use Gap-up Strategy", value=True)
        gap_up_min_ratio = st.slider("Gap-up Minimum Ratio (Open / Prev High)", 1.0, 1.10, 1.02)

        st.markdown("---")
        st.subheader("Market Selection")

        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sectors = sorted(sp500["GICS Sector"].unique().tolist())
        industries = sorted(sp500["GICS Sub-Industry"].unique().tolist())

        market = st.radio("Choose market", ["S&P 500", "Small Caps", "Upload Custom CSV"])

        if market == "S&P 500":
            sector_filter = st.multiselect("Filter by Sector", sectors)
            industry_filter = st.multiselect("Filter by Industry", industries)
            filtered_sp500 = sp500
            if sector_filter:
                filtered_sp500 = filtered_sp500[filtered_sp500["GICS Sector"].isin(sector_filter)]
            if industry_filter:
                filtered_sp500 = filtered_sp500[filtered_sp500["GICS Sub-Industry"].isin(industry_filter)]
            symbols = filtered_sp500["Symbol"].tolist()
        elif market == "Small Caps":
            symbols = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]
        else:
            uploaded_file = st.file_uploader("Upload CSV with one ticker per line")
            symbols = []
            if uploaded_file:
                try:
                    user_df = pd.read_csv(uploaded_file, header=None)
                    symbols = user_df[0].dropna().unique().tolist()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        selected = st.multiselect("Select stocks to scan (leave empty to scan all)", symbols)

        if st.button("🔍 Scan Now"):
            tickers = selected if selected else symbols
            st.info(f"Scanning {len(tickers)} stocks...")

            params = {
                "use_rsi": use_rsi,
                "rsi_threshold": rsi_threshold,
                "use_min_volume": use_min_volume,
                "min_volume": min_volume,
                "use_breakout": use_breakout,
                "use_volume_spike": use_volume_spike,
                "volume_spike_multiplier": volume_spike_multiplier,
                "use_gap_up": use_gap_up,
                "gap_up_min_ratio": gap_up_min_ratio,
            }

            results = []
            for ticker in tickers:
                res = scan_stock(ticker, params)
                if res:
                    results.append(res)

            if results:
                df = pd.DataFrame(results)
                st.success(f"Found {len(df)} matches")
                st.dataframe(df)
                st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")

                # Email alert option
                send_email = st.checkbox("Send Email Alert with results")
                if send_email:
                    to_email = st.text_input("To Email")
                    from_email = st.text_input("From Email (Gmail)")
                    email_password = st.text_input("Email Password (App Password)", type="password")
                    if st.button("Send Email"):
                        subject = f"Stock Scanner Alert - {len(df)} Matches"
                        body = df.to_string()
                        send_email_alert(subject, body, to_email, from_email, email_password)

                # Interactive Chart
                st.markdown("---")
                st.subheader("Stock Price Chart")
                chart_ticker = st.selectbox("Select ticker to view chart", options=df["Ticker"].tolist())
                if chart_ticker:
                    stock_data = yf.Ticker(chart_ticker).history(period="1mo", interval="1h")
                    if stock_data is not None and not stock_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=stock_data.index,
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name=chart_ticker
                        ))
                        fig.update_layout(title=f"{chart_ticker} Last Month Hourly Prices", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No chart data available for this ticker")
            else:
                st.warning("No stocks matched the criteria.")

        if enable_debug:
            st.markdown("---")
            st.subheader("Debug Logs")
            debug_logs = st.session_state.get("debug_log", [])
            for log in debug_logs:
                st.text(log)


if __name__ == "__main__":
    main()
