import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Strategy Scanner", layout="centered")
st.title("📈 Stock Strategy Scanner")
st.caption("Screen S&P 500 stocks with custom strategies like RSI, breakouts, volume spikes, and gap-ups.")

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    return table[0]['Symbol'].tolist()

tickers = get_sp500_tickers()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

st.sidebar.header("📌 Strategy Filters")
enable_breakout = st.sidebar.checkbox("Breakout above 20-day high")
enable_volume_spike = st.sidebar.checkbox("Volume spike (2× avg)")
enable_gap_up = st.sidebar.checkbox("Gap-up (>2% over prev high)")

def scan():
    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, period='2mo', interval='1d', progress=False)
            if len(data) < 20:
                continue

            data['RSI'] = calculate_rsi(data)
            latest = data.iloc[-1]
            prev = data.iloc[-2]

            rsi = latest['RSI']
            vol = latest['Volume']
            price = latest['Close']

            if pd.isna(rsi) or rsi >= 30 or vol < 1_000_000:
                continue

            if enable_breakout:
                recent_high = data['High'].rolling(20).max().iloc[-2]
                if price <= recent_high:
                    continue

            if enable_volume_spike:
                avg_vol = data['Volume'].rolling(20).mean().iloc[-2]
                if vol <= 2 * avg_vol:
                    continue

            if enable_gap_up:
                if latest['Open'] <= prev['High'] * 1.02:
                    continue

            results.append({
                'Ticker': ticker,
                'Price': round(price, 2),
                'RSI': round(rsi, 2),
                'Volume': int(vol)
            })

        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")

        progress.progress((i + 1) / len(tickers))

    return pd.DataFrame(results).sort_values(by='RSI')

if st.button("🔍 Run Scan"):
    st.write("Scanning S&P 500 stocks...")
    df = scan()

    if not df.empty:
        st.success(f"Found {len(df)} matching stocks.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "stock_scan_results.csv", "text/csv")
    else:
        st.warning("No matching stocks found.")
