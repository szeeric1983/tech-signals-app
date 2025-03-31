
import streamlit as st
import pandas as pd
import requests
import ta
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 技術指標信號分析（改良邏輯：買入多於賣出）")

import streamlit as st
POLYGON_API_KEY = st.secrets["api_keys"]["POLYGON_API_KEY"]

if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False
if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = ""

def trigger_analysis(symbol=None):
    if symbol:
        st.session_state["selected_symbol"] = symbol
    st.session_state["run_analysis"] = True

default_symbols = ["TSLA", "AAPL", "TLT", "NVDA", "TQQQ", "SQQQ", "GOOG", "META", "TEM"]
st.write("📌 快速選擇股票：")
cols = st.columns(len(default_symbols))
for i, sym in enumerate(default_symbols):
    if cols[i].button(sym):
        trigger_analysis(sym)

symbol_input = st.text_input("或輸入其他股票代碼", key="symbol_input", on_change=trigger_analysis)

st.markdown("""
### 📋 使用的 5 個技術指標（買賣對立，優先判斷佔多數）：

| 指標 | 買入條件 ✅ | 賣出條件 ❌ | 類別 |
|------|-------------|--------------|------|
| **RSI** | RSI < 30 | RSI > 70 | 動能 |
| **MACD** | MACD > Signal | MACD < Signal | 趨勢轉向 |
| **MA20** | 接近 MA20 (±3%) | 高於 MA20 5% | 趨勢偏離 |
| **Volume Spike** | 成交量 > 20 日均量 x2 | 無 | 成交異常 |
| **Bollinger Bands** | 跌穿下限 | 升穿上限 | 波動率擴張 |
""", unsafe_allow_html=True)

def get_polygon_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=365 * 3)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "results" not in data:
        return None
    df = pd.DataFrame(data["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={"c": "Close", "v": "Volume"}, inplace=True)
    return df

def add_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi().fillna(0)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd().fillna(0)
    df['MACD_signal'] = macd.macd_signal().fillna(0)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband().fillna(0)
    df['BB_lower'] = bb.bollinger_lband().fillna(0)
    return df

def evaluate_signals(row):
    buy, sell, neutral = [], [], []

    if row['RSI'] < 30:
        buy.append("RSI<30")
    elif row['RSI'] > 70:
        sell.append("RSI>70")
    if row['MACD'] > row['MACD_signal']:
        buy.append("MACD+")
    elif row['MACD'] < row['MACD_signal']:
        sell.append("MACD−")
    if abs(row['Close'] - row['MA20']) / row['Close'] < 0.03:
        buy.append("接近MA20")
    elif (row['Close'] - row['MA20']) / row['Close'] > 0.05:
        sell.append("高於MA20")
    if row['Volume'] > 2 * row['Vol_MA20']:
        buy.append("Volume Spike")
    if row['Close'] < row['BB_lower']:
        buy.append("跌穿Boll下限")
    elif row['Close'] > row['BB_upper']:
        sell.append("升穿Boll上限")

    return buy, sell

def find_signals(df):
    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        buy, sell = evaluate_signals(row)
        total = len(buy) + len(sell)
        if total >= 3:
            if len(buy) > len(sell):
                signal_type = "Buy"
            elif len(sell) > len(buy):
                signal_type = "Sell"
            else:
                signal_type = ""
            if signal_type:
                entry_price = row["Close"]
                result = {
                    "Date": row.name.date(),
                    "Close": round(entry_price, 2),
                    "符合指標數": total,
                    "中咗邊幾個指標": "，".join(buy + sell),
                    "Signal": signal_type
                }
                for label, days in {
                    "1 Week": 5,
                    "2 Weeks": 10,
                    "1 Month": 21,
                    "3 Months": 63
                }.items():
                    future_idx = i + days
                    if future_idx < len(df):
                        future_price = df["Close"].iloc[future_idx]
                        pct_return = (future_price - entry_price) / entry_price * 100
                        result[label + " Price"] = round(future_price, 2)
                        result[label + " Return"] = f"{round(pct_return, 2)}%"
                        correct = "✅" if (signal_type == "Buy" and pct_return > 0) or (signal_type == "Sell" and pct_return < 0) else "❌"
                        result[label + " 判斷"] = correct
                    else:
                        result[label + " Price"] = "N/A"
                        result[label + " Return"] = "N/A"
                        result[label + " 判斷"] = "-"
                records.append(result)
    return pd.DataFrame(records)

def signal_summary(df, label):
    if df.empty:
        return "⚠️ 無紀錄"
    corrects = df[[c for c in df.columns if "判斷" in c]].apply(lambda row: sum(cell == "✅" for cell in row), axis=1)
    summary = f"📌 共 {len(df)} 次 {label} Signal，正確次數（至少 1 時段）✅：{sum(corrects > 0)} 次（命中率 {round(100 * sum(corrects > 0)/len(df), 2)}%）"
    return summary

final_symbol = st.session_state["selected_symbol"] or symbol_input

if st.session_state["run_analysis"] and final_symbol:
    with st.spinner(f"分析中：{final_symbol}..."):
        df = get_polygon_data(final_symbol)
        if df is None or df.empty:
            st.error("❌ 無法獲取資料，請檢查代碼是否正確")
        else:
            df = add_indicators(df)
            full_df = find_signals(df)
            if full_df.empty:
                st.warning("⚠️ 沒有符合條件的 signal")
            else:
                buy_df = full_df[full_df["Signal"] == "Buy"].reset_index(drop=True)
                sell_df = full_df[full_df["Signal"] == "Sell"].reset_index(drop=True)
                st.subheader("📥 Buy Signal")
                st.markdown(signal_summary(buy_df, "Buy"))
                st.dataframe(buy_df)

                st.subheader("📤 Sell Signal")
                st.markdown(signal_summary(sell_df, "Sell"))
                st.dataframe(sell_df)

# 新增功能：Buy/Sell 信號摘要 + 最終建議
def summarize_signals(signals):
    buy_signals = signals[0]
    sell_signals = signals[1]
    summary = f"📌 買入信號 ({len(buy_signals)}): " + ", ".join(buy_signals) + "\n"
    summary += f"📌 賣出信號 ({len(sell_signals)}): " + ", ".join(sell_signals)
    decision = "Buy ✅" if len(buy_signals) > len(sell_signals) else ("Sell ❌" if len(sell_signals) > len(buy_signals) else "中性 ⚪️")
    return summary, decision

# 最近一次明確訊號
def find_recent_signal(df):
    for i in reversed(range(len(df))):
        row = df.iloc[i]
        signals = evaluate_signals(row)
        if len(signals[0]) > len(signals[1]):
            return df.index[i].date(), "Buy ✅"
        elif len(signals[1]) > len(signals[0]):
            return df.index[i].date(), "Sell ❌"
    return None, "中性 ⚪️"

# 最近一週價格與成交量變化總結
def show_week_summary(df):
    recent_week = df[-7:]
    price_change = (recent_week["Close"].iloc[-1] - recent_week["Close"].iloc[0]) / recent_week["Close"].iloc[0] * 100
    volume_change = (recent_week["Volume"].iloc[-1] - recent_week["Volume"].iloc[0]) / recent_week["Volume"].iloc[0] * 100
    rsi_avg = recent_week["RSI"].mean()

    daily_returns = recent_week["Close"].pct_change().dropna()
    max_gain_day = daily_returns.idxmax().date()
    max_gain = daily_returns.max() * 100
    max_loss_day = daily_returns.idxmin().date()
    max_loss = daily_returns.min() * 100

    macd_cross = "無明顯交叉"
    for i in range(len(recent_week)-1, 0, -1):
        if recent_week["MACD"].iloc[i] > recent_week["MACD_signal"].iloc[i] and recent_week["MACD"].iloc[i-1] <= recent_week["MACD_signal"].iloc[i-1]:
            macd_cross = f"黃金交叉 ({recent_week.index[i].date()})"
            break
        elif recent_week["MACD"].iloc[i] < recent_week["MACD_signal"].iloc[i] and recent_week["MACD"].iloc[i-1] >= recent_week["MACD_signal"].iloc[i-1]:
            macd_cross = f"死亡交叉 ({recent_week.index[i].date()})"
            break

    bb_upper_touches = (recent_week["Close"] > recent_week["BB_upper"]).sum()
    bb_lower_touches = (recent_week["Close"] < recent_week["BB_lower"]).sum()

    summary_text = f"""
📆 最近一週技術分析 ({recent_week.index[0].date()} ~ {recent_week.index[-1].date()}):
- 價格變動：{price_change:.2f}%
- 成交量變動：{volume_change:.2f}%
- RSI 平均值：{rsi_avg:.2f}（>70超買, <30超賣）
- 最大單日升幅：{max_gain:.2f}% ({max_gain_day})
- 最大單日跌幅：{max_loss:.2f}% ({max_loss_day})
- MACD 最近交叉：{macd_cross}
- 布林帶觸碰次數：上軸 {bb_upper_touches} 次，下軸 {bb_lower_touches} 次

📚 指標解釋：
- **RSI 平均值**：一週內平均相對強弱指數，>70為超買（可能會回調），<30為超賣（可能會反彈）
- **單日升跌幅**：每日收市價變化的最大升幅及跌幅
- **MACD 最近交叉**：最近一次趨勢轉折，黃金交叉代表可能轉升，死亡交叉代表可能轉跌
- **布林帶觸碰次數**：價格碰到布林帶上限或下限，代表短線價格波動較大
"""
    return summary_text
    recent_week = df[-7:]
    price_change = (recent_week["Close"].iloc[-1] - recent_week["Close"].iloc[0]) / recent_week["Close"].iloc[0] * 100
    volume_change = (recent_week["Volume"].iloc[-1] - recent_week["Volume"].iloc[0]) / recent_week["Volume"].iloc[0] * 100
    return f"📅 最近一週價格變動：{price_change:.2f}%\n📊 成交量變動：{volume_change:.2f}%"

# 主程式加上新功能
if st.session_state["run_analysis"]:
    symbol = st.session_state["selected_symbol"] or symbol_input.upper()
    df = get_polygon_data(symbol)
    if df is not None:
        df = add_indicators(df)
        latest_row = df.iloc[-1]
        signals = evaluate_signals(latest_row)
        
        # 信號摘要
        summary, decision = summarize_signals(signals)
        st.subheader(f"📊 {symbol} 信號摘要與建議")
        st.write(summary)
        st.write(f"最終建議：**{decision}**")

        # 最近一次明確訊號
        recent_date, recent_signal = find_recent_signal(df)
        if recent_date:
            st.info(f"🕓 最近一次明確訊號：{recent_signal} (日期：{recent_date})")
        else:
            st.info("🕓 最近沒有明確訊號")

        # 最近一週總結
        week_summary = show_week_summary(df)
        st.success(week_summary)
    else:
        st.error("無法獲取股票資料，請檢查輸入或API連接。")
