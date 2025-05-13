
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# ✅ This must come FIRST before any Streamlit UI commands
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")

    st.subheader("🎛️ Filter Presets")
    preset = st.selectbox("Choose Preset", ["Custom", "Conservative", "Aggressive"])

    # Default values
    rsi_threshold = 30
    min_volume = 1_000_000
    vol_spike_ratio = 2.0
    gap_up_percent = 2.0

    if preset == "Conservative":
        rsi_threshold = 40
        min_volume = 3_000_000
        vol_spike_ratio = 2.5
        gap_up_percent = 3.0
    elif preset == "Aggressive":
        rsi_threshold = 60
        min_volume = 500_000
        vol_spike_ratio = 1.5
        gap_up_percent = 1.0

    if preset == "Custom":
        st.subheader("🔧 Manual Settings")
        rsi_threshold = st.slider("Max RSI (0 to disable)", 0, 100, rsi_threshold)
        min_volume = st.number_input("Minimum Volume", value=min_volume, step=100_000)
        vol_spike_ratio = st.slider("Volume Spike Ratio", 1.0, 5.0, vol_spike_ratio, step=0.1)
        gap_up_percent = st.slider("Gap-Up % over previous high", 0.0, 10.0, gap_up_percent, step=0.1)

    debug_mode = st.checkbox("🔍 Enable Debug Mode (show exclusions)", value=False)

st.title("📈 Stock Strategy Scanner")

# --- Stock scanning logic ---

def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            return None, f"{ticker}: Not enough data"

        latest = data.iloc[-1]
        close = latest['Close']
        high = latest['High']
        open_ = latest['Open']
        vol = latest['Volume']

        # RSI
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        if rsi_threshold > 0 and (pd.isna(latest_rsi) or latest_rsi > rsi_threshold):
            return None, f"{ticker}: RSI {latest_rsi:.2f} > {rsi_threshold}"

        if vol < min_volume:
            return None, f"{ticker}: Volume {vol:,} < {min_volume:,}"

        day20_high = data['High'].rolling(20).max()
        if high < day20_high.iloc[-2]:
            return None, f"{ticker}: High {high:.2f} < 20-day high {day20_high.iloc[-2]:.2f}"

        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= vol_spike_ratio * avg_vol.iloc[-2]:
            return None, f"{ticker}: Volume {vol:,} ≤ {vol_spike_ratio} × Avg {avg_vol.iloc[-2]:,.0f}"

        prev_high = data['High'].iloc[-2]
        gap_target = (1 + gap_up_percent / 100) * prev_high
        if open_ < gap_target:
            return None, f"{ticker}: Open {open_:.2f} < Gap target {gap_target:.2f}"

        return {
            "Ticker": ticker,
            "Close": round(close, 2),
            "Volume": int(vol),
            "RSI": round(latest_rsi, 2),
            "Breakout": True,
            "Volume Spike": True,
            "Gap Up": True,
        }, None

    except Exception as e:
        return None, f"{ticker}: Error - {e}"

# --- Source symbols ---
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
symbols_500 = sp500["Symbol"].tolist()

small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

st.subheader("Select Market Segment")

market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])
available_symbols = []

if market == "S&P 500":
    available_symbols = symbols_500
elif market == "Small Caps":
    available_symbols = small_caps
else:
    uploaded_file = st.file_uploader("Upload CSV with one ticker per line")
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file, header=None)
            available_symbols = user_df[0].dropna().unique().tolist()
        except:
            st.error("⚠️ Error reading the uploaded file. Make sure it's a CSV with one column of tickers.")

# --- UI: Scan button ---
if available_symbols:
    selected = st.multiselect("Select stocks to scan (or leave empty to scan all)", available_symbols)

    if st.button("🔍 Scan Now"):
        tickers = selected if selected else available_symbols
        st.info(f"Scanning {len(tickers)} stocks...")

        results = []
        excluded = []

        for sym in tickers:
            result, reason = scan_stock(sym)
            if result:
                results.append(result)
            elif debug_mode:
                excluded.append({"Ticker": sym, "Reason": reason})

        if results:
            df = pd.DataFrame(results)
            st.success(f"✅ Found {len(df)} matching stocks")
            st.dataframe(df)
            st.download_button("📥 Download Matches", df.to_csv(index=False), "scanner_matches.csv")

        else:
            st.warning("❌ No stocks matched the current strategy filters.")

        if debug_mode and excluded:
            st.markdown("### ⚠️ Excluded Stocks and Reasons")
            excluded_df = pd.DataFrame(excluded)
            st.dataframe(excluded_df)
            st.download_button("📤 Download Excluded", excluded_df.to_csv(index=False), "scanner_excluded.csv")

