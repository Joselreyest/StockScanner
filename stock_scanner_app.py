
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# ✅ This must come FIRST before any Streamlit UI commands
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")
st.title("📈 Stock Strategy Scanner")

def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
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

        print(f"{ticker}: RSI={latest_rsi:.2f}, Vol={vol}")
        
        if pd.isna(latest_rsi) or latest_rsi > 30:
            print(f"{ticker} excluded: RSI > 30")
            return None
            
        if vol < 1_000_000:
            print(f"{ticker} excluded: volume < 1M")
            return None

        # Breakout
        day20_high = data['High'].rolling(20).max()
        if high < day20_high.iloc[-2]:
            print(f"{ticker} excluded: not a 20-day breakout")
            return None

        # Volume spike
        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= 2 * avg_vol.iloc[-2]:
            print(f"{ticker} excluded: no volume spike")
            return None

        # Gap-up
        prev_high = data['High'].iloc[-2]
        if open_ < 1.02 * prev_high:
            print(f"{ticker} excluded: no gap up")
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
        print(f"Error fetching {ticker}: {e}")
        return None

sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
symbols_500 = sp500["Symbol"].tolist()

small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

st.subheader("Select Market Segment")

market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])

symbols = []
uploaded_file = None

if market == "S&P 500":
    symbols = symbols_500
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
        for sym in tickers:
            res = scan_stock(sym)
            if res:
                results.append(res)

        if results:
            df = pd.DataFrame(results)
            st.success(f"Found {len(df)} matches")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "scanner_results.csv")
        else:
            st.warning("No stocks matched the criteria.")

