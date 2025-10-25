# -------------------------------------------------------
# 📈 STOCK PRICE PREDICTION USING MACHINE LEARNING (UI)
# -------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# 🧠 Helper Function – MACD Calculation
# -------------------------------------------------------
def calc_macd(data, len1, len2, len3):
    shortEMA = data.ewm(span=len1, adjust=False).mean()
    longEMA = data.ewm(span=len2, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=len3, adjust=False).mean()
    return MACD, signal


# -------------------------------------------------------
# ⚙️ Core Function – Train Model and Predict Prices
# -------------------------------------------------------
def train_and_predict(ticker, years):
    # Fetch data from Yahoo Finance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    history = yf.download(ticker, start=start_date, end=end_date, interval='1d', prepost=False)

    # 🔧 FIX for multi-index columns
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [col[0] for col in history.columns]

    # Check for empty data
    if history.empty:
        st.error("⚠️ No data found for this symbol. Try another one.")
        return None

    # Keep necessary columns
    history = history.loc[:, ['Open', 'Close', 'Volume']]

    # Feature Engineering
    history['Prev_Close'] = history['Close'].shift(1)
    history['Prev_Volume'] = history['Volume'].shift(1)
    history['weekday'] = history.index.weekday

    # Simple Moving Averages (SMAs)
    for sma in [5, 10, 20, 50, 100, 200]:
        history[f'{sma}SMA'] = history['Prev_Close'].rolling(sma).mean()

    # MACD Indicators
    MACD, signal = calc_macd(history['Prev_Close'], 12, 26, 9)
    history['MACD'] = MACD
    history['MACD_signal'] = signal

    # Clean data
    history = history.replace(np.inf, np.nan).dropna()

    # Train-Test Split
    X = history.drop(['Close', 'Volume'], axis=1).values
    y = history['Close']
    num_test = 365

    X_train, y_train = X[:-num_test], y[:-num_test]
    X_test, y_test = X[-num_test:], y[-num_test:]

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return history, y_test, preds


# -------------------------------------------------------
# 💰 Helper Function – Profit Simulation
# -------------------------------------------------------
def simulate_investment(opens, closes, preds, start_account=1000, thresh=0):
    account = start_account
    changes = []

    for i in range(len(preds)):
        if (preds[i] - opens[i]) / opens[i] >= thresh:
            account += account * (closes[i] - opens[i]) / opens[i]
        changes.append(account)

    changes = np.array(changes)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(changes)), changes, color='green')
    ax.set_title("Investment Growth Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)

    invest_total = start_account + start_account * (closes[-1] - opens[0]) / opens[0]
    st.write(f"💵 **Normal Investing:** ${invest_total:.2f} ({((invest_total - start_account) / start_account * 100):.1f}%)")
    st.write(f"🤖 **Model-based Trading:** ${account:.2f} ({((account - start_account) / start_account * 100):.1f}%)")


# -------------------------------------------------------
# 🌐 STREAMLIT APP UI
# -------------------------------------------------------
st.set_page_config(page_title="📊 Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction using Machine Learning")

st.markdown(
    """
    This app uses **Linear Regression** to predict stock prices based on historical data 
    fetched from **Yahoo Finance** and displays various technical indicators like 
    *Moving Averages* and *MACD*.  
    """
)

# --- User Inputs ---
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
years = st.slider("Select number of years of data:", 1, 15, 5)

if st.button("🔮 Predict Stock Prices"):
    with st.spinner("Fetching data and training model... ⏳"):
        result = train_and_predict(ticker, years)

    if result:
        history, y_test, preds = result

        # --- Price Trend Chart ---
        st.subheader(f"📅 {ticker} Stock Trend ({years} Years)")
        st.line_chart(history[['Close', '5SMA', '10SMA', '50SMA', '200SMA']])

        # --- Prediction vs Actual ---
        st.subheader("📈 Model Prediction vs Actual Prices (Last 1 Year)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(y_test)), y_test, label='Actual', color='blue')
        ax.plot(range(len(preds)), preds, label='Predicted', color='red')
        ax.legend()
        st.pyplot(fig)

        # --- Profit Simulation ---
        st.subheader("💰 Investment Simulation")
        simulate_investment(history['Open'][-len(y_test):].values,
                            y_test.values, preds, 1000, 0)

        st.success("✅ Prediction and simulation completed!")


st.markdown("---")
st.caption("Made with ❤️ using Streamlit, scikit-learn, and yfinance")
