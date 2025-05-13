import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import ta  # technical analysis indicators

# ✅ This must come FIRST before any Streamlit UI commands
st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("**Stock Strategy Scanner**")
    st.caption("by Jose Reyes")

st.title("📈 Stock Strategy Scanner")

# User-configurable strategy filters
st.sidebar.header("🔧 Strategy Filters")

min_volume = st.sidebar.slider("Minimum Volume", 100_000, 10_000_000, 1_000_000, step=100_000)
max_rsi = st.sidebar.slider("Max RSI", 10, 70, 30)
gap_up_pct = st.sidebar.slider("Min Gap % (Open vs Prev High)", 0.5, 10.0, 2.0, step=0.5)
volume_spike_ratio = st.sidebar.slider("Min Volume Spike Ratio", 1.0, 10.0, 2.0, step=0.5)

# Load S&P 500 with sectors
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
sp500 = sp500[['Symbol', 'Security', 'GICS Sector']]
sp500.columns = ['Symbol', 'Company', 'Sector']
symbols_500 = sp500['Symbol'].tolist()

# Define small caps (example)
small_caps = ["PLUG", "FUBO", "BB", "NNDM", "GPRO", "AMC", "CLSK", "MARA", "RIOT", "SOUN"]

# Market selector
st.subheader("Select Market Segment")
market = st.radio("Choose market type", ["S&P 500", "Small Caps", "Upload Custom CSV"])

symbols = []
uploaded_file = None

if market == "S&P 500":
    sectors = ['All'] + sorted(sp500['Sector'].unique())
    selected_sector = st.selectbox("Filter by Sector", sectors)

    if selected_sector != 'All':
        filtered_df = sp500[sp500['Sector'] == selected_sector]
        symbols = filtered_df['Symbol'].tolist()
    else:
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
            st.error("⚠️ Error reading uploaded file. Ensure it’s a single-column CSV.")

# Stock scan logic
def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo', interval='1d')

        if data is None or data.empty or len(data) < 21:
            return None

        latest = data.iloc[-1]
        close = latest['Close']
        high = latest['High']
        low = latest['Low']
        open_ = latest['Open']
        vol = latest['Volume']

        # RSI using ta
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        latest_rsi = data['RSI'].iloc[-1]
        if pd.isna(latest_rsi) or latest_rsi > max_rsi:
            return None

        if vol < min_volume:
            return None

        # Breakout logic
        day20_high = data['High'].rolling(window=20).max()
        if high < day20_high.iloc[-2]:
            return None

        # Volume spike logic
        avg_vol = data['Volume'].rolling(20).mean()
        if pd.isna(avg_vol.iloc[-2]) or vol <= volume_spike_ratio * avg_vol.iloc[-2]:
            return None

        # Gap-up logic
        prev_high = data['High'].iloc[-2]
        if open_ < (1 + gap_up_pct / 100) * prev_high:
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

# UI for stock selection
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

            # Chart viewer
            selected_ticker = st.selectbox("📊 View chart for:", df["Ticker"].tolist())
            if selected_ticker:
                st.subheader(f"{selected_ticker} - Candlestick + RSI")
                chart_data = yf.Ticker(selected_ticker).history(period="2mo", interval="1d")
                chart_data['RSI'] = ta.momentum.RSIIndicator(chart_data['Close']).rsi()

                fig = go.Figure()

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name="Candlestick"
                ))

                # RSI
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data["RSI"],
                    mode='lines',
                    name='RSI',
                    yaxis='y2',
                    line=dict(color='orange')
                ))

                # Layout
                fig.update_layout(
                    height=600,
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(title="Price"),
                    yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
                    title=f"{selected_ticker} Price & RSI"
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stocks matched the criteria.")
