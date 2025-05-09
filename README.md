# 📈 Stock Strategy Scanner

A customizable stock scanner built with **Streamlit** and **Yahoo Finance**, designed to help you identify high-potential stocks in the **S&P 500** using technical strategies.

![App Screenshot](https://raw.githubusercontent.com/yourusername/your-repo-name/main/screenshot.png) <!-- Optional: Add your app screenshot here -->

---

## 🚀 Features

- ✅ RSI Scanner (RSI < 30 + Volume > 1M)
- 📊 Optional filters:
  - Breakout above 20-day high
  - Volume spikes (2× average)
  - Gap-ups (>2% over previous high)
- 🗂️ Download results as CSV
- ⚡ Fast scanning of the full S&P 500
- 🌐 Deployable on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🔧 Setup & Run Locally

```bash
git clone https://github.com/yourusername/stock-strategy-scanner.git
cd stock-strategy-scanner
pip install -r requirements.txt
streamlit run stock_scanner_app.py
```

---

## 📸 Screenshot

<!-- Replace this with a real screenshot -->
![Scanner UI](https://via.placeholder.com/800x400?text=App+Screenshot)

---

## 📚 Technologies

- Streamlit
- yFinance
- pandas
- numpy
- lxml

---

## ☁️ Deploy on Streamlit Cloud

1. Push this project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "New App"
3. Select your repo and `stock_scanner_app.py`
4. Click **Deploy**

---

## 🙌 Contributing

Pull requests welcome! Add strategies, UI improvements, or performance enhancements.

---

## 📬 Contact

Created by Jose Reyes  
📧 joselreyest@gmail.com 
🔗 [LinkedIn](https://linkedin.com/in/jreyest)

---

## 📝 License

Open source under the [MIT License](LICENSE).
