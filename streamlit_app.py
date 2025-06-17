import os
import time
import threading
import json
import ssl
from email.message import EmailMessage

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import schedule
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize NLTK VADER lexicon
try:
    _ = nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Stock Strategy Scanner", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar: Filters & Settings
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Scanner Settings")

st.session_state.volume_spike_factor = st.sidebar.slider(
    "Volume Spike Multiplier", 1.0, 5.0, 2.0, step=0.1
)
st.session_state.gap_up_factor = st.sidebar.slider(
    "Gap Up Factor", 1.00, 1.20, 1.02, step=0.01
)
st.session_state.rsi_max = st.sidebar.slider("Max RSI", 10, 100, 70)
st.session_state.min_volume = st.sidebar.number_input(
    "Min Volume", min_value=0, value=100000
)
st.session_state.min_sentiment_score = st.sidebar.slider(
    "Minimum Sentiment Score", -1.0, 1.0, 0.0, step=0.05
)
st.session_state.sentiment_weight = st.sidebar.slider(
    "Sentiment Weight", 0.0, 1.0, 0.2, step=0.05
)

st.session_state.alert_email = st.sidebar.text_input(
    "Alert Email (optional)"
)
st.session_state.enable_debug = st.sidebar.checkbox("Enable Debug Log")

index_choice = st.sidebar.selectbox(
    "Choose Universe", ["NASDAQ", "S&P 500", "Small Caps"]
)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (one ticker per row)", type=["csv"]
)

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def log_debug(msg: str):
    if st.session_state.get("enable_debug", False):
        st.session_state.debug_log = st.session_state.get("debug_log", []) + [msg]
        try:
            print("DEBUG:", msg)
        except:
            pass

@st.cache_data
def get_nasdaq_symbols():
    url = "https://api.nasdaq.com/api/screener/stocks?exchange=nasdaq&download=true"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = requests.get(url, headers=headers).json()
    return [row["symbol"] for row in data["data"]["rows"]]

@st.cache_data
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    df = pd.read_html(html, attrs={"id": "constituents"})[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()

@st.cache_data
def get_small_cap_symbols():
    return ["AVXL", "PLTR", "BB", "MVIS", "NNDM", "HIMS"]

@st.cache_data
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def compute_features_for_universe(symbols):
    rows = []
    for sym in symbols:
        hist = yf.Ticker(sym).history(period="60d", interval="1d")
        if len(hist) < 15:
            continue
        rsi = compute_rsi(hist["Close"]).iloc[-1]
        vol_spike = hist["Volume"].iloc[-1] / hist["Volume"].rolling(20).mean().iloc[-2]
        mom_1d = hist["Close"].pct_change(1).iloc[-1]
        mom_5d = hist["Close"].pct_change(5).iloc[-1]
        sentiment = get_news_sentiment(sym)
        rows.append({
            "Symbol": sym,
            "rsi": rsi,
            "vol_spike": vol_spike,
            "mom_1d": mom_1d,
            "mom_5d": mom_5d,
            "sentiment": sentiment
        })
    df = pd.DataFrame(rows).set_index("Symbol")
    scaler = StandardScaler()
    df = df.dropna()
    if df.empty:
        raise ValueError("No valid feature rows to process after dropping NaNs.")
    df.loc[:, :] = scaler.fit_transform(df)
    return df

def format_email_table(df: pd.DataFrame) -> str:
    style = """
    <style>
      table {border-collapse: collapse; width: 100%;}
      th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
      th {background: #4CAF50; color: white;}
      tr:nth-child(even){background: #f2f2f2;}
    </style>
    """
    return style + df.to_html(index=False, escape=False)

def send_email_alert(subject: str, body: str = None, html: str = None):
    receiver = st.session_state.get("alert_email", "")
    if not receiver:
        log_debug("No alert email configured; skipping email.")
        return
    msg = EmailMessage()
    if html:
        msg.set_content("This email contains HTML content.")
        msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(body or "No data available.")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("EMAIL_USER")
    msg["To"] = receiver
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
            server.send_message(msg)
        log_debug(f"Email sent to {receiver}")
    except Exception as e:
        log_debug(f"Failed to send email: {e}")

def get_news_sentiment(symbol: str) -> float:
    try:
        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            return 0.0
        url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize=5&apiKey={api_key}"
        articles = requests.get(url).json().get("articles", [])
        if not articles:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(a["title"] + ". " + a.get("description",""))["compound"]
                  for a in articles]
        avg = float(np.mean(scores))
        log_debug(f"{symbol} sentiment: {avg:.3f}")
        return avg
    except Exception as e:
        log_debug(f"Sentiment error for {symbol}: {e}")
        return 0.0

def plot_chart(symbol: str):
    df = yf.Ticker(symbol).history(period="1mo", interval="1d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", opacity=0.4
    ), secondary_y=True)
    fig.update_layout(title=f"{symbol} Price & Volume", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}")

# Load ML model for jump prediction
@st.cache_resource
def load_jump_model():
    return joblib.load("models/intraday_jump_model.pkl")

model = load_jump_model()

# ────────────────────────────────────────────────────────────────────────────────
# Main App Logic
# ────────────────────────────────────────────────────────────────────────────────
st.title("📈 Stock Strategy Scanner & Jump Predictor")

# Determine universe
if uploaded_file:
    df_u = pd.read_csv(uploaded_file, header=None)
    symbols = df_u[0].dropna().astype(str).tolist()
elif index_choice == "NASDAQ":
    symbols = get_nasdaq_symbols()
elif index_choice == "S&P 500":
    symbols = get_sp500_symbols()
else:
    symbols = get_small_cap_symbols()

# Button: run feature-based ML jump prediction
if st.button("▶️ Score Universe for Jump Probability"):
    feats = compute_features_for_universe(symbols)
    if feats.empty:
        st.warning("No symbols with sufficient history.")
    else:
        probs = model.predict_proba(feats)[:,1]
        preds = pd.Series(probs, index=feats.index, name="JumpProbability")
        top_n = st.sidebar.number_input("Show Top N", 1, 50, 10)
        top = preds.nlargest(top_n).to_frame()
        top["JumpProbability"] = top["JumpProbability"].round(3)
        st.subheader(f"Top {top_n} Jump Candidates")
        st.dataframe(top)
        st.bar_chart(top["JumpProbability"])

        # Optionally email results
        if st.session_state.alert_email:
            html = format_email_table(top.reset_index().rename(columns={"index":"Symbol"}))
            send_email_alert("Jump Candidates", html=html)

        # Chart drill‐down
        pick = st.selectbox("Chart:", top.index.tolist())
        plot_chart(pick)

# Debug log
if st.session_state.enable_debug:
    with st.expander("🔍 Debug Log", expanded=False):
        for entry in st.session_state.get("debug_log", []):
            st.text(entry)
