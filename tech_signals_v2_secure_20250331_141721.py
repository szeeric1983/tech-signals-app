
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import xgboost as xgb
from prophet import Prophet
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split

API_KEY = "LPCXqapazYAqjuuFRq62mpoiyce66wJ2"
today = datetime.date.today()
from_date = today - datetime.timedelta(days=730)

# 股票選擇 UI
categories = {
    "MAG7": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
    "AI 股": ["NVDA", "AMD", "SMCI", "TSM", "ASML"],
    "ETF": ["SPY", "QQQ", "TQQQ", "SQQQ", "ARKK"],
    "黃金 / 債券 / 金融叵": ["GLD", "TLT", "XLF", "JPM", "BAC"]
}

st.sidebar.header("📊 股票選擇")
selected_symbol = st.sidebar.text_input("🔍 搜尋股票代碼", "AAPL")
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 熱門分類快速選擇")

for group in categories:
    st.sidebar.markdown(f"**{group}**")
    for sym in categories[group]:
        if st.sidebar.button(sym, key=f"btn_{group}_{sym}"):
            selected_symbol = sym

symbol = selected_symbol
st.title(f"📈 {symbol} 技術分析 + AI 預測系統")

@st.cache_data
def fetch_stock_data(symbol, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={API_KEY}"
    r = requests.get(url)
    data = r.json().get("results", [])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"t": "Date", "c": "Close", "v": "Volume"})
    df = df[["Date", "Close", "Volume"]].dropna()
    return df

df = fetch_stock_data(symbol, from_date, today)
if df.empty:
    st.error("❌ 無法取得資料")
    st.stop()

# 技術指標
macd = MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

rsi = RSIIndicator(close=df["Close"])
df["RSI"] = rsi.rsi()

bb = BollingerBands(close=df["Close"])
df["bb_upper"] = bb.bollinger_hband()
df["bb_lower"] = bb.bollinger_lband()
df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

df["volume_avg20"] = df["Volume"].rolling(20).mean()
df["volume_spike"] = df["Volume"] > df["volume_avg20"] * 1.5

df["TD_Buy_Count"] = 0
df["TD_Sell_Count"] = 0
for i in range(4, len(df)):
    df.loc[df.index[i], "TD_Buy_Count"] = df["Close"].iloc[i] < df["Close"].iloc[i - 4]
    df.loc[df.index[i], "TD_Sell_Count"] = df["Close"].iloc[i] > df["Close"].iloc[i - 4]
df["TD_Buy_Count"] = df["TD_Buy_Count"].rolling(9).sum()
df["TD_Sell_Count"] = df["TD_Sell_Count"].rolling(9).sum()

latest = df.iloc[-1]

# Prophet 預測
st.subheader("🔮 使用 Prophet 預測未來 7 日價格")
prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
m = Prophet(daily_seasonality=True)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
fig1 = m.plot(forecast)
st.pyplot(fig1)

latest_forecast = forecast.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
st.markdown("### 📅 預測價格（未來 7 日）")
st.dataframe(latest_forecast.rename(columns={
    "ds": "日期", "yhat": "預測價格", "yhat_lower": "下限", "yhat_upper": "上限"
}).set_index("日期").style.format("{:.2f}"))

# XGBoost 模型
st.subheader("🧠 AI 模型預測（XGBoost）")
df["target"] = df["Close"].shift(-3) > df["Close"]
features = ["MACD_diff", "RSI", "bb_pct", "volume_spike", "TD_Buy_Count"]
df_model = df[features + ["target"]].dropna()
df_model["volume_spike"] = df_model["volume_spike"].astype(int)
X = df_model[features]
y = df_model["target"].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

latest_features = df[features].iloc[-1:]
latest_features["volume_spike"] = latest_features["volume_spike"].astype(int)
proba = model.predict_proba(latest_features)[0][1]
buy_prob = proba * 100

if buy_prob >= 80:
    rating = "🌟 極度買入"
elif buy_prob >= 65:
    rating = "✅ 買入"
elif buy_prob >= 45:
    rating = "🟡 中性"
elif buy_prob >= 30:
    rating = "⚠️ 賣出"
else:
    rating = "🚨 極度賣出"

st.metric("📊 預測買入機率", f"{buy_prob:.2f}%", rating)

# 條形圖 + 解釋
weights = {
    "MACD_diff": latest_features["MACD_diff"].values[0] * 3,
    "RSI": (70 - abs(latest_features["RSI"].values[0] - 50)) / 10,
    "BB 百分比": latest_features["bb_pct"].values[0] * 2,
    "Volume Spike": latest_features["volume_spike"].values[0] * 2,
    "TD 九轉": latest_features["TD_Buy_Count"].values[0]
}
st.bar_chart(pd.DataFrame.from_dict(weights, orient="index", columns=["加權分數"]))
st.dataframe(pd.DataFrame.from_dict(weights, orient="index", columns=["加權分數"]))

# 自動解釋區
st.markdown("### 🤖 自動技術分析解釋")
analysis = []

macd_val = latest_features["MACD_diff"].values[0]
if macd_val > 1:
    analysis.append(f"MACD 差距為 {macd_val:.2f}，屬於強勢趨勢信號。")
elif macd_val > 0:
    analysis.append(f"MACD 差距為 {macd_val:.2f}，初步顯示多頭傾向。")
else:
    analysis.append(f"MACD 差距為 {macd_val:.2f}，尚未形成黃金交叉。")

rsi_val = latest_features["RSI"].values[0]
if rsi_val > 70:
    analysis.append(f"RSI 為 {rsi_val:.1f}，已進入過熱區，需留意回調風險。")
elif rsi_val > 55:
    analysis.append(f"RSI 為 {rsi_val:.1f}，處於偏強區間。")
elif rsi_val > 45:
    analysis.append(f"RSI 為 {rsi_val:.1f}，接近中性。")
else:
    analysis.append(f"RSI 為 {rsi_val:.1f}，偏弱或接近超賣區。")

bb_pct_val = latest_features["bb_pct"].values[0]
if bb_pct_val > 0.85:
    analysis.append(f"Bollinger Bands 顯示價格接近上軌，需留意可能回落。")
elif bb_pct_val < 0.15:
    analysis.append(f"Bollinger Bands 顯示價格接近下軌，有反彈空間。")
else:
    analysis.append(f"價格處於 Bollinger Band 區間中段，屬健康波動。")

vol_spike = latest_features["volume_spike"].values[0]
if vol_spike:
    analysis.append("今日出現 Volume Spike，可能為突破訊號。")
else:
    analysis.append("今日未見明顯成交量異常。")

td_val = latest_features["TD_Buy_Count"].values[0]
if td_val == 9:
    analysis.append("TD 九轉買入訊號已完成（第 9 棒）。")
elif td_val >= 6:
    analysis.append(f"TD 九轉已推進至第 {int(td_val)} 棒，接近買入完成。")
else:
    analysis.append(f"TD 九轉目前為第 {int(td_val)} 棒，仍需觀察。")

if rating.startswith("🌟"):
    summary = "整體技術面非常強勁，屬於理想買入時機。"
elif rating.startswith("✅"):
    summary = "技術面整體偏多，可考慮分批進場。"
elif rating.startswith("🟡"):
    summary = "部分指標偏多，但仍需觀察確認。"
elif rating.startswith("⚠️"):
    summary = "多項指標轉弱，建議保持觀望。"
else:
    summary = "技術面明顯轉弱，建議減持或暫避風險。"

for line in analysis:
    st.write("• " + line)
st.markdown(f"**📌 綜合結論：{summary}**")
