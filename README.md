# 📈 Stock Strategy Scanner

A powerful Streamlit-based app that scans stocks based on customizable filters like RSI, volume spike, and gap-ups. Ideal for day traders and swing traders seeking momentum plays.

---

## 🚀 Features

- 🧠 RSI filter, volume spike detection, and gap-up breakout strategy
- 📈 Interactive chart with SMA, Bollinger Bands, RSI, and MACD
- 📬 Optional email alerts
- 📅 Daily scan scheduler (runs while app is open)
- 🧪 Debug mode with real-time logging
- 💾 Save & load filter presets
- 📁 Upload custom ticker list via CSV

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

To enable email alerts, set the following secrets on [Streamlit Cloud](https://streamlit.io/cloud):

```toml
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_gmail_app_password"
```

> Tip: Use a [Gmail App Password](https://support.google.com/accounts/answer/185833?hl=en) if you have 2FA enabled.

---

## 📁 File Structure

```
📦 stock-scanner-app
├── app.py
├── requirements.txt
├── logo.png
├── filter_presets.json (optional)
└── README.md
```

---

## 📄 How to Run Locally

```bash
streamlit run app.py
```

---

## 🧪 Example Filters

- **RSI < 70**
- **Volume > 100K**
- **Gap up > 2%**
- **Volume Spike > 1.5x previous**

---

## 📬 Example Email Alert

```
Subject: Stock Scanner Alert

Symbol   RSI   Volume   Gap Up   Score
AAPL     65.2  2300000  True     3
NVDA     58.1  1800000  True     3
```

---

## 🙌 Author

Created by **Jose Reyes**  
Feel free to fork, customize, and share!
