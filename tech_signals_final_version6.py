import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from prophet import Prophet
import yfinance as yf
import ta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import logging
import atexit
import shutil
from pmdarima import auto_arima

# 設置日誌
logging.basicConfig(filename='error_log.txt', level=logging.INFO)

# 清理 cmdstanpy 臨時檔案
def cleanup_cmdstanpy_tmpdir():
    try:
        shutil.rmtree(Prophet()._tmpdir, ignore_errors=True)
    except Exception as e:
        logging.error(f"清理 cmdstanpy 臨時檔案失敗: {e}")

atexit.register(cleanup_cmdstanpy_tmpdir)

# Streamlit 頁面配置
st.set_page_config(page_title="智能選股平台 PRO (美股版)", layout="wide", initial_sidebar_state="expanded")

# 使用說明
st.sidebar.title("📘 使用說明")
show_tech_help = st.sidebar.checkbox("📊 顯示技術指標說明", value=False)
if show_tech_help:
    st.sidebar.markdown("""
    #### 🟢 技術指標說明 (美股優化版)
    
    **初階指標升級：**
    - 📈 **智能RSI 2.0**：科技股<25/其他<30為超賣，>75/>70為超買，需ATR確認
    - 📉 **MACD 2.0**：黃金交叉需連續兩日價格+ATR確認
    - 🎚️ **布林通道PRO**：突破需>1.5倍ATR，結合VWAP
    - 💹 **成交量進階**：科技股需2.5倍均量/其他2倍，配合OBV
    
    **高階指標新增：**
    - 📏 **VWAP**：成交量加權平均價，價格>VWAP為多頭信號
    - 🌪️ **ATR過濾**：確保突破信號真實，>1.5倍ATR才觸發
    - 🌀 **TD9結構+**：完成後需成交量+ATR配合
    - 🔄 **RSI背離2.0**：價格波動>5%且ATR確認
    
    **預測模型升級：**
    - ⏳ **Prophet 30天預測**：藍色線為實際價格，綠色線為預測價格，淺藍色為置信區間，納入納斯達克100指數
    - 🚀 **XGBoost/隨機森林/LightGBM進階**：調優參數，新增52週高低點與納斯達克相關性，特徵重要性可選顯示
    - 📉 **ARIMA**：提供線性時間序列預測，補充 Prophet 的長期趨勢分析
    
    **美股專屬：**
    - 🏷️ **行業自適應**：科技股更寬鬆閾值
    - ⚖️ **多空權重**：多頭市場信號放大1.2x
    - 📐 **黃金比率回調**：關鍵支持阻力位
    - 🖼️ **圖形形態**：頭肩頂/底、雙頂/底，10天內有效
    """)

# 股票分類（包含中文名稱）
stock_categories = {
    "熱門股": [
        ("TSLA", "特斯拉"), ("NVDA", "英偉達"), ("TQQQ", "三倍做多納指ETF"),
        ("SQQQ", "三倍做空納指ETF"), ("TEM", "Tempus AI"), ("AAPL", "蘋果"),
        ("TLT", "20年期國債ETF")
    ],
    "MAG7": [
        ("AAPL", "蘋果"), ("MSFT", "微軟"), ("GOOG", "谷歌"), ("AMZN", "亞馬遜"),
        ("META", "Meta"), ("NVDA", "英偉達"), ("TSLA", "特斯拉")
    ],
    "AI 股": [
        ("NVDA", "英偉達"), ("AMD", "超微"), ("TSM", "台積電"), ("PLTR", "Palantir"),
        ("SMCI", "超微電腦"), ("ASML", "阿斯麥"), ("TEM", "Tempus AI"), ("INTC", "英特爾"),
        ("SNOW", "Snowflake")
    ],
    "ETF": [
        ("SPY", "標普500 ETF"), ("QQQ", "納斯達克100 ETF"), ("TQQQ", "三倍做多納指ETF"),
        ("SQQQ", "三倍做空納指ETF"), ("ARKK", "方舟創新ETF"), ("VOO", "先鋒標普500 ETF"),
        ("IWF", "羅素1000成長ETF")
    ],
    "黃金 / 債券 / 金融": [
        ("GLD", "黃金ETF"), ("TLT", "20年期國債ETF"), ("XLF", "金融板塊ETF"),
        ("JPM", "摩根大通"), ("BAC", "美國銀行"), ("SLV", "白銀ETF"), ("GS", "高盛")
    ],
    "比特幣 / 區塊鏈": [
        ("BITO", "比特幣期貨ETF"), ("MARA", "馬拉松數位"), ("RIOT", "Riot Platforms"),
        ("COIN", "Coinbase"), ("GBTC", "灰度比特幣信託"), ("ETHE", "灰度以太坊信託")
    ],
    "能源 / 石油": [
        ("XLE", "能源板塊ETF"), ("CVX", "雪佛龍"), ("XOM", "埃克森美孚"),
        ("OXY", "西方石油"), ("BP", "英國石油"), ("SLB", "斯倫貝謝")
    ],
    "5G / 半導體": [
        ("QCOM", "高通"), ("AVGO", "博通"), ("SWKS", "思佳訊"), ("NXPI", "恩智浦"),
        ("MRVL", "邁威爾科技"), ("AMD", "超微")
    ],
    "新能源 / 電動車": [
        ("LI", "理想汽車"), ("NIO", "蔚來"), ("LCID", "Lucid"), ("RIVN", "Rivian"),
        ("FSLR", "第一太陽能")
    ],
    "醫療 / 生物科技": [
        ("PFE", "輝瑞"), ("MRNA", "Moderna"), ("GILD", "吉利德科學"), ("BIIB", "百健")
    ],
    "看淡槓桿 ETF": [
        ("NVDQ", "2倍做空英偉達ETF"), ("NVD", "1.5倍做空英偉達ETF"), ("NVD3", "3倍做空英偉達ETF"),
        ("TSLS", "做空特斯拉ETF"), ("TSLZ", "2倍做空特斯拉ETF"), ("TSLQ", "1.5倍做空特斯拉ETF"),
        ("3STP", "3倍做空標普科技ETF")
    ]
}

# 數據獲取函數
def fetch_price_data(symbol, retries=5):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365)
    
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date + dt.timedelta(days=1), interval="1d")
            fetch_time = dt.datetime.now()
            
            if df.empty or df["Close"].isna().all():
                raise ValueError(f"yfinance: 無有效數據 for {symbol}")
            
            df = df.reset_index().rename(columns={
                "Date": "date",
                "Close": "close",
                "Volume": "volume",
                "High": "high",
                "Low": "low"
            })
            df["date"] = df["date"].dt.tz_localize(None)
            df = df[["date", "close", "volume", "high", "low"]]
            df = df.dropna()  # 移除無效數據
            
            logging.info(f"yfinance: 數據獲取成功 for {symbol} at {fetch_time}, latest date: {df['date'].max()}")
            return df, fetch_time, "yfinance"
        except Exception as e:
            logging.error(f"yfinance fetch_price_data({symbol}) 失敗 on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                st.error(f"無法獲取 {symbol} 的數據，請檢查網絡或股票代碼")
                return None, None, None

def fetch_nasdaq_data():
    df, fetch_time, source = fetch_price_data("QQQ")
    if df is not None:
        df = df.rename(columns={"close": "nasdaq_close", "volume": "nasdaq_volume"})
        return df[["date", "nasdaq_close", "nasdaq_volume"]]
    return None

# 技術指標計算
def calculate_all_indicators(df, sector=""):
    df = df.copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    
    # 確保無 NaN 或零值
    df = df.dropna()
    if len(df) < 50:  # 確保足夠數據
        logging.warning(f"數據不足 for {sector}: {len(df)} rows")
        return df, {}, [], []
    
    df["ma5"] = pd.Series(closes).rolling(window=5).mean()
    df["ma20"] = pd.Series(closes).rolling(window=20).mean()
    df["ma50"] = pd.Series(closes).rolling(window=50).mean()
    df["rsi"] = ta.momentum.RSIIndicator(pd.Series(closes)).rsi()
    macd = ta.trend.MACD(pd.Series(closes))
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(pd.Series(closes))
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_break"] = df.apply(
        lambda row: "上軌" if row.close > row.bb_upper else ("下軌" if row.close < row.bb_lower else "中間"), 
        axis=1
    )
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_signal"] = df["close"] > df["vwap"]
    df["atr"] = ta.volatility.AverageTrueRange(pd.Series(highs), pd.Series(lows), pd.Series(closes)).average_true_range()
    df["vol_avg20"] = pd.Series(volumes).rolling(window=20).mean()
    vol_threshold = 2.5 if sector in ["Technology", "Consumer Cyclical"] else 2
    df["vol_spike"] = df["volume"] > vol_threshold * df["vol_avg20"]
    df["adx"] = ta.trend.ADXIndicator(pd.Series(highs), pd.Series(lows), pd.Series(closes)).adx()
    df["td9_down"] = (df["close"] < df["close"].shift(4)).astype(int)
    df["td9_count"] = df["td9_down"] * (
        df["td9_down"].groupby((df["td9_down"] != df["td9_down"].shift()).cumsum()).cumcount() + 1
    )
    obv = [0]
    for i in range(1, len(df)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_ma20"] = pd.Series(obv).rolling(window=20).mean()
    df["52w_high"] = pd.Series(highs).rolling(window=252, min_periods=20).max()
    df["52w_low"] = pd.Series(lows).rolling(window=252, min_periods=20).min()
    
    lookback = 60
    recent_high = pd.Series(highs[-lookback:]).max()
    recent_low = pd.Series(lows[-lookback:]).min()
    diff = recent_high - recent_low
    fib_levels = {
        "0%": recent_high,
        "23.6%": recent_high - diff * 0.236,
        "38.2%": recent_high - diff * 0.382,
        "50%": recent_high - diff * 0.5,
        "61.8%": recent_high - diff * 0.618,
        "100%": recent_low
    }
    
    hs_patterns = []
    dt_patterns = []
    for i in range(2, len(df)-2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
            highs[i-1] > highs[i-2] and highs[i+1] > highs[i+2]):
            hs_patterns.append(("頭肩頂", df["date"].iloc[i]))
        if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
            lows[i-1] < lows[i-2] and lows[i+1] < lows[i+2]):
            hs_patterns.append(("頭肩底", df["date"].iloc[i]))
    for i in range(1, len(df)-1):
        threshold = 0.05
        if (abs(highs[i] - highs[i-1]) < threshold * highs[i] and
            highs[i] > highs[i-2] and highs[i] > highs[i+1]):
            dt_patterns.append(("雙頂", df["date"].iloc[i]))
        if (abs(lows[i] - lows[i-1]) < threshold * lows[i] and
            lows[i] < lows[i-2] and lows[i] < lows[i+1]):
            dt_patterns.append(("雙底", df["date"].iloc[i]))
    
    return df, fib_levels, hs_patterns, dt_patterns

# 信號生成
def enhanced_generate_signal(df, fib_levels, hs_patterns, dt_patterns, sector=""):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 2 else prev
    score = 0
    explanation = []
    weight = 1.2 if latest["close"] > df["ma50"].iloc[-1] else 0.8
    
    def price_confirmation(condition, periods=2):
        if periods == 2:
            return condition(latest["close"], prev["close"]) and condition(prev["close"], prev2["close"])
        return condition(latest["close"], prev["close"])
    
    rsi_buy_threshold = 25 if sector in ["Technology", "Consumer Cyclical"] else 30
    rsi_sell_threshold = 75 if sector in ["Technology", "Consumer Cyclical"] else 70
    if latest.rsi < rsi_buy_threshold and price_confirmation(lambda x, y: x > y) and (latest["close"] - prev["close"]) > 1.5 * latest["atr"]:
        score += 1.5 * weight
        explanation.append(f"RSI超賣({latest.rsi:.1f})，價格+ATR確認")
    elif latest.rsi > rsi_sell_threshold and price_confirmation(lambda x, y: x < y) and (prev["close"] - latest["close"]) > 1.5 * latest["atr"]:
        score -= 1 * weight
        explanation.append(f"RSI超買({latest.rsi:.1f})，價格+ATR確認")
    
    if latest.macd_hist > 0 and prev.macd_hist < 0 and price_confirmation(lambda x, y: x > y) and (latest["close"] - prev["close"]) > 1.5 * latest["atr"]:
        score += 1.2 * weight
        explanation.append(f"MACD柱狀圖翻正，價格+ATR確認")
    elif latest.macd_hist < 0 and prev.macd_hist > 0 and price_confirmation(lambda x, y: x < y):
        score -= 1 * weight
        explanation.append(f"MACD柱狀圖翻負，價格確認")
    
    if latest.bb_break == "下軌" and (latest["close"] - latest["bb_lower"]) > 1.5 * latest["atr"]:
        score += 1.5 * weight
        explanation.append(f"布林帶跌破下軌，ATR確認")
    elif latest.bb_break == "上軌" and (latest["bb_upper"] - latest["close"]) > 1.5 * latest["atr"]:
        score -= 1 * weight
        explanation.append(f"布林帶突破上軌，ATR確認")
    
    if latest.vwap_signal and price_confirmation(lambda x, y: x > y):
        score += 1.2 * weight
        explanation.append(f"價格突破VWAP({latest.vwap:.2f})，多頭信號")
    elif not latest.vwap_signal and price_confirmation(lambda x, y: x < y):
        score -= 0.8 * weight
        explanation.append(f"價格跌破VWAP({latest.vwap:.2f})，空頭信號")
    
    if latest.vol_spike and latest.close > prev.close:
        score += 1.8 * weight
        explanation.append(f"放量上漲，成交量:{latest.volume/1e6:.1f}M")
    
    if latest.adx > 25:
        score += 0.5 * weight if latest["close"] > df["ma50"].iloc[-1] else -0.5 * weight
        explanation.append(f"{'強勁上升' if latest['close'] > df['ma50'].iloc[-1] else '強勁下降'}趨勢(ADX:{latest.adx:.1f})")
    
    if df["td9_count"].iloc[-1] >= 9 and latest.vol_spike:
        score += 1.5 * weight
        explanation.append(f"TD9結構完成，成交量確認")
    
    if latest.obv > latest.obv_ma20 and latest.close > prev.close:
        score += 0.8 * weight
        explanation.append(f"OBV突破20日均線，價格上升")
    
    current_price = latest["close"]
    for level, price in fib_levels.items():
        if abs(current_price - price) < current_price * 0.01:
            if level in ["38.2%", "50%", "61.8%"]:
                score += 1
                explanation.append(f"觸及黃金比率 {level} ({price:.2f})")
    
    if hs_patterns and (df["date"].iloc[-1] - hs_patterns[-1][1]).days < 10:
        last_pattern = hs_patterns[-1]
        score += 1.5 if last_pattern[0] == "頭肩底" else -1.5
        explanation.append(f"近期{last_pattern[0]} (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
    if dt_patterns and (df["date"].iloc[-1] - dt_patterns[-1][1]).days < 10:
        last_pattern = dt_patterns[-1]
        score += 1 if last_pattern[0] == "雙底" else -1
        explanation.append(f"近期{last_pattern[0]} (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
    
    signal = (
        "🟢 強力買入" if score >= 5 else
        "🟡 謹慎買入" if score >= 3 else
        "🔴 強力賣出" if score <= -3 else
        "🔴 考慮賣出" if score <= -1 else
        "⚪ 中性觀望"
    )
    
    return signal, explanation, score, latest

# 圖表
def plot_fibonacci_levels(fig, df, fib_levels):
    last_date = df["date"].iloc[-1]
    for level, price in fib_levels.items():
        fig.add_shape(
            type="line", x0=df["date"].iloc[0], y0=price, x1=last_date, y1=price,
            line=dict(color="purple", width=1, dash="dot"), name=f"黃金比率 {level}"
        )
        fig.add_annotation(x=last_date, y=price, text=f"黃金比率 {level}", showarrow=False, yshift=10)
    return fig

def plot_ichimoku(fig, df):
    high_9 = df["high"].rolling(window=9).max()
    low_9 = df["low"].rolling(window=9).min()
    df["ichi_tenkan"] = (high_9 + low_9) / 2
    high_26 = df["high"].rolling(window=26).max()
    low_26 = df["low"].rolling(window=26).min()
    df["ichi_kijun"] = (high_26 + low_26) / 2
    df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(26)
    high_52 = df["high"].rolling(window=52).max()
    low_52 = df["low"].rolling(window=52).min()
    df["ichi_senkou_b"] = ((high_52 + low_52) / 2).shift(26)
    
    fig.add_trace(go.Scatter(x=df["date"], y=df["ichi_senkou_a"], line=dict(width=0), name="雲層上緣"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ichi_senkou_b"], line=dict(width=0), name="雲層下緣", fill="tonexty", fillcolor="rgba(100,100,255,0.2)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ichi_tenkan"], line=dict(color="green", width=1), name="轉換線"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ichi_kijun"], line=dict(color="red", width=1), name="基準線"))
    return fig

# 單一股票分析
def analyze_single_stock(symbol, sector=""):
    try:
        df, fetch_time, source = fetch_price_data(symbol)
        if df is None or df.empty:
            logging.error(f"數據獲取失敗 for {symbol}")
            return None, None, None, None, None, None, None, None, None, None, None
        
        df, fib_levels, hs_patterns, dt_patterns = calculate_all_indicators(df, sector)
        signal, explanation, score, latest = enhanced_generate_signal(df, fib_levels, hs_patterns, dt_patterns, sector)
        
        # Prophet 預測
        forecast_trend = "N/A"
        current_price = None
        forecast = None
        df_prophet = None
        try:
            df_prophet = df[["date", "close", "volume"]].copy()
            df_prophet = df_prophet.rename(columns={"date": "ds", "close": "y"})
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
            
            nasdaq_df = fetch_nasdaq_data()
            if nasdaq_df is not None:
                df_prophet = df_prophet.merge(
                    nasdaq_df[["date", "nasdaq_close", "nasdaq_volume"]], 
                    left_on="ds", right_on="date", how="left"
                ).drop(columns=["date"])
                df_prophet["nasdaq_close"] = df_prophet["nasdaq_close"].ffill()
                df_prophet["nasdaq_volume"] = df_prophet["nasdaq_volume"].ffill()
            
            m = Prophet(daily_seasonality=True)
            if nasdaq_df is not None:
                m.add_regressor("nasdaq_close")
                m.add_regressor("nasdaq_volume")
                m.add_regressor("volume")
            
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=30)
            
            if nasdaq_df is not None:
                future = future.merge(
                    nasdaq_df[["date", "nasdaq_close", "nasdaq_volume"]], 
                    left_on="ds", right_on="date", how="left"
                ).drop(columns=["date"])
                future["nasdaq_close"] = future["nasdaq_close"].ffill()
                future["nasdaq_volume"] = future["nasdaq_volume"].ffill()
                future["volume"] = df_prophet["volume"].iloc[-1]
            
            forecast = m.predict(future)
            forecast_mean = forecast.tail(30)["yhat"].mean()
            current_price = df_prophet["y"].iloc[-1]
            forecast_trend = "上漲" if forecast_mean > current_price else "下跌"
        except Exception as e:
            logging.error(f"Prophet 預測失敗 for {symbol}: {e}")
        
        # ARIMA 預測
        arima_trend = "N/A"
        arima_forecast = None
        try:
            if df_prophet is not None:
                arima_model = auto_arima(df_prophet["y"], seasonal=True, m=7, suppress_warnings=True)
                arima_forecast = arima_model.predict(n_periods=30)
                arima_trend = "上漲" if arima_forecast.mean() > current_price else "下跌"
        except Exception as e:
            logging.error(f"ARIMA 預測失敗 for {symbol}: {e}")
        
        # XGBoost、隨機森林和 LightGBM 預測
        model_results = []
        up_proba = None
        rf_proba = None
        lgbm_proba = None
        xgb_model = None
        try:
            df_xgb = df.copy()
            if len(df_xgb) < 100:
                up_proba = None
                rf_proba = None
                lgbm_proba = None
                logging.warning(f"數據不足 for ML models: {len(df_xgb)} rows")
            else:
                df_xgb["return"] = df_xgb["close"].pct_change()
                df_xgb["target"] = (df_xgb["return"].shift(-1) > 0).astype(int)
                df_xgb["ma5"] = df_xgb["close"].rolling(5).mean()
                df_xgb["ma20"] = df_xgb["close"].rolling(20).mean()
                df_xgb["rsi"] = ta.momentum.RSIIndicator(df_xgb["close"]).rsi()
                df_xgb["macd"] = ta.trend.MACD(df_xgb["close"]).macd_diff()
                df_xgb["adx"] = ta.trend.ADXIndicator(df_xgb["high"], df_xgb["low"], df_xgb["close"]).adx()
                df_xgb["vol"] = df_xgb["volume"]
                df_xgb["obv"] = df_xgb["obv"]
                df_xgb["52w_high"] = df_xgb["52w_high"]
                df_xgb["52w_low"] = df_xgb["52w_low"]
                
                nasdaq_df = fetch_nasdaq_data()
                if nasdaq_df is not None:
                    df_xgb = df_xgb.merge(nasdaq_df[["date", "nasdaq_close"]], on="date", how="left")
                    df_xgb["nasdaq_close"] = df_xgb["nasdaq_close"].ffill()
                    df_xgb["nasdaq_corr"] = df_xgb["close"].rolling(20, min_periods=10).corr(df_xgb["nasdaq_close"])
                
                df_xgb = df_xgb.dropna().copy()
                if df_xgb.empty:
                    up_proba = None
                    rf_proba = None
                    lgbm_proba = None
                    logging.warning(f"數據清理後為空 for {symbol}")
                else:
                    features = ["ma5", "ma20", "rsi", "macd", "adx", "vol", "obv", "52w_high", "52w_low"]
                    if nasdaq_df is not None and "nasdaq_corr" in df_xgb.columns:
                        features.append("nasdaq_corr")
                    
                    X = df_xgb[features]
                    y = df_xgb["target"]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
                    
                    # XGBoost
                    xgb_model = GridSearchCV(XGBClassifier(eval_metric="logloss"), 
                                             param_grid={"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}, 
                                             cv=5)
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    xgb_acc = accuracy_score(y_test, xgb_pred)
                    xgb_proba = xgb_model.best_estimator_.predict_proba(X_scaled[-1:])[0][1]
                    model_results.append({"模型": "XGBoost", "上升機率 (%)": xgb_proba*100, "測試集準確率 (%)": xgb_acc*100})
                    up_proba = xgb_proba
                    
                    # 隨機森林
                    rf_model = GridSearchCV(RandomForestClassifier(random_state=42), 
                                            param_grid={"n_estimators": [100, 200], "max_depth": [3, 5, 7]}, 
                                            cv=5)
                    rf_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    rf_acc = accuracy_score(y_test, rf_pred)
                    rf_proba = rf_model.best_estimator_.predict_proba(X_scaled[-1:])[0][1]
                    model_results.append({"模型": "Random Forest", "上升機率 (%)": rf_proba*100, "測試集準確率 (%)": rf_acc*100})
                    
                    # LightGBM
                    lgbm_model = GridSearchCV(LGBMClassifier(random_state=42), 
                                              param_grid={"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}, 
                                              cv=5)
                    lgbm_model.fit(X_train, y_train)
                    lgbm_pred = lgbm_model.predict(X_test)
                    lgbm_acc = accuracy_score(y_test, lgbm_pred)
                    lgbm_proba = lgbm_model.best_estimator_.predict_proba(X_scaled[-1:])[0][1]
                    model_results.append({"模型": "LightGBM", "上升機率 (%)": lgbm_proba*100, "測試集準確率 (%)": lgbm_acc*100})
        except Exception as e:
            logging.error(f"分類模型預測失敗 for {symbol}: {e}")
        
        return df, forecast_trend, (up_proba, rf_proba, lgbm_proba), (signal, explanation, score, latest), fetch_time, source, arima_forecast, model_results, arima_trend, forecast, df_prophet
    except Exception as e:
        logging.error(f"analyze_single_stock({symbol}) failed: {e}")
        return None, None, None, None, None, None, None, None, None, None, None

# 主應用
st.title("📈 智能選股平台 PRO (美股版)")

st.sidebar.header("📂 股票分類")
category = st.sidebar.selectbox("選擇分類：", list(stock_categories.keys()))
stock_list = stock_categories[category]

selected_symbol = None
for stock, name in stock_list:
    label = f"{stock} - {name}"
    if st.sidebar.button(label, key=f"btn_{stock}"):
        selected_symbol = stock

custom = st.sidebar.text_input("或輸入股票代碼：", "")
if custom:
    custom = custom.upper()
    if custom.strip() and custom.isalnum():  # 驗證輸入
        selected_symbol = custom
    else:
        st.sidebar.error("請輸入有效的股票代碼")

if selected_symbol:
    try:
        ticker = yf.Ticker(selected_symbol)
        sector = ticker.info.get("sector", "未知")
    except Exception as e:
        logging.error(f"獲取 {selected_symbol} sector 失敗: {e}")
        sector = "未知"
        st.warning(f"無法獲取 {selected_symbol} 的行業資訊，使用預設值 '未知'")

    result = analyze_single_stock(selected_symbol, sector)
    
    if len(result) == 11:
        df, forecast_trend, probas, signal_data, fetch_time, source, arima_forecast, model_results, arima_trend, forecast, df_prophet = result
    else:
        df, forecast_trend, probas, signal_data, fetch_time, source, arima_forecast, model_results, arima_trend, forecast, df_prophet = None, None, None, None, None, None, None, None, None, None, None
    
    if df is not None and not df.empty and signal_data is not None:
        signal, explanation, score, latest = signal_data
        latest_price = df["close"].iloc[-1]
        up_proba, rf_proba, lgbm_proba = probas if probas else (None, None, None)
        
        # 顯示股票名稱、最新股價和數據來源
        st.subheader(f"🔍 分析股票：{selected_symbol} - {signal}")
        st.markdown(f"💰 **最新股價**：${latest_price:.2f}")
        st.markdown(f"📌 **綜合評分**：{score:.1f}/10")
        st.markdown(f"📡 **數據來源**：{source}（取得時間：{fetch_time}）")
        
        # 檢查股價是否合理
        if selected_symbol == "TSLA" and abs(latest_price - 237.97) > 5:
            st.warning(f"⚠️ 股價可能未更新，TSLA 應接近 $237.97（2025-04-22 收盤價）")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("趨勢強度", f"{latest.adx:.1f}", "強勢" if latest.adx > 25 else "弱勢", delta_color="inverse")
        with col2:
            st.metric("市場方向", "多頭" if latest["close"] > df["ma50"].iloc[-1] else "空頭", f"{latest.close/df['ma50'].iloc[-1]-1:.2%}")
        with col3:
            st.metric("行業特性", sector, "高波動" if sector in ["Technology", "Consumer Cyclical"] else "一般")
        
        with st.expander("📋 詳細訊號分析", expanded=True):
            st.markdown("### 當前市場狀態")
            ma50_trend = "上升" if latest["close"] > df["ma50"].iloc[-1] else "下降"
            st.write(f"- **趨勢方向**: {ma50_trend} (收盤價 {latest['close']:.2f} vs MA50 {df['ma50'].iloc[-1]:.2f})")
            st.write(f"- **VWAP信號**: {'多頭' if latest.vwap_signal else '空頭'} (價格 {latest.close:.2f} vs VWAP {latest.vwap:.2f})")
            st.write(f"- **趨勢強度**: {'強勁' if latest.adx > 25 else '普通'} (ADX值:{latest.adx:.1f})")
            
            st.markdown("### 技術指標分析")
            if not explanation:
                st.warning("⚠️ 當前無明顯技術信號")
            else:
                for item in explanation:
                    if "買" in item or "漲" in item:
                        st.success(f"✅ {item}")
                    elif "賣" in item or "跌" in item:
                        st.error(f"❌ {item}")
                    else:
                        st.info(f"ℹ️ {item}")
        
        st.markdown("---")
        st.subheader("📊 進階技術指標分析")
        fib_levels, hs_patterns, dt_patterns = calculate_all_indicators(df, sector)[1:4]
        st.markdown("### 黃金比率回調位")
        fib_table = pd.DataFrame.from_dict(fib_levels, orient="index", columns=["價位"])
        st.table(fib_table.style.format({"價位": "{:.2f}"}))
        
        if hs_patterns or dt_patterns:
            st.markdown("### 🔄 圖形形態")
            col1, col2 = st.columns(2)
            with col1:
                if hs_patterns:
                    st.write("#### 頭肩形態")
                    st.dataframe(pd.DataFrame(hs_patterns, columns=["形態", "日期"]).tail(3))
            with col2:
                if dt_patterns:
                    st.write("#### 雙頂/雙底")
                    st.dataframe(pd.DataFrame(dt_patterns, columns=["形態", "日期"]).tail(3))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="收盤價", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(dash="dot", color="#ff7f0e")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], name="MA50", line=dict(dash="dash", color="#2ca02c")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["vwap"], name="VWAP", line=dict(color="#9467bd")))
        fig = plot_fibonacci_levels(fig, df, fib_levels)
        fig = plot_ichimoku(fig, df)
        fig.update_layout(title=f"{selected_symbol} 技術分析圖", xaxis_title="日期", yaxis_title="價格", legend=dict(orientation="h"), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prophet 和 ARIMA 預測
        with st.expander("⏳ AI 模組：Prophet & ARIMA 預測未來 30 日", expanded=True):
            if forecast is None or df_prophet is None:
                st.error("❌ Prophet 預測失敗，無法顯示預測圖表")
            else:
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="實際價格", line=dict(color="#1f77b4")))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet 預測", line=dict(color="#2ca02c")))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), fillcolor="rgba(31,119,180,0.2)", showlegend=False))
                    
                    # 添加 ARIMA 預測
                    if arima_forecast is not None:
                        forecast_dates = pd.date_range(start=df["date"].iloc[-1] + dt.timedelta(days=1), periods=30)
                        fig.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast, mode="lines", name="ARIMA 預測", line=dict(color="#ff7f0e")))
                    
                    fig.update_layout(title="Prophet & ARIMA 預測未來 30 日收盤價", xaxis_title="日期", yaxis_title="價格")
                    
                    forecast_next30 = forecast.tail(30).copy()
                    forecast_next30.loc[:, "預測日"] = [f"T+{i+1}" for i in range(30)]
                    forecast_table = forecast_next30[["預測日", "ds", "yhat", "yhat_lower", "yhat_upper"]]
                    forecast_table.columns = ["預測日", "日期", "Prophet 預測收盤價", "最低估價", "最高估價"]
                    if arima_forecast is not None:
                        forecast_table.loc[:, "ARIMA 預測收盤價"] = arima_forecast
                    
                    st.subheader("📋 Prophet & ARIMA 預測未來 30 日價格詳情")
                    st.dataframe(
                        forecast_table.style.format({
                            "Prophet 預測收盤價": "{:.2f}", 
                            "最低估價": "{:.2f}", 
                            "最高估價": "{:.2f}",
                            "ARIMA 預測收盤價": "{:.2f}" if arima_forecast is not None else None
                        })
                    )
                    st.markdown("📈 **預測解讀**: 藍色線為實際價格，綠色線為 Prophet 預測，橙色線為 ARIMA 預測，淺藍色為 Prophet 的 80% 置信區間")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 繪製 Prophet & ARIMA 預測圖表失敗：{e}")
        
        # XGBoost、隨機森林和 LightGBM 預測
        with st.expander("🚀 AI 模組：XGBoost, Random Forest, LightGBM 升跌預測", expanded=True):
            if up_proba is None and rf_proba is None and lgbm_proba is None:
                st.warning("⚠️ 數據不足（少於100天）或處理失敗，無法進行漲跌預測")
            else:
                st.write(f"📈 **XGBoost 預測下一日上升機率**：{up_proba*100:.2f}%")
                st.write(f"📈 **Random Forest 預測下一日上升機率**：{rf_proba*100:.2f}%")
                st.write(f"📈 **LightGBM 預測下一日上升機率**：{lgbm_proba*100:.2f}%")
                
                # 顯示模型比較表格
                st.write("### 模型比較")
                comparison = pd.DataFrame(model_results)
                st.dataframe(comparison.style.format({"上升機率 (%)": "{:.2f}", "測試集準確率 (%)": "{:.2f}"}))

                if st.checkbox("顯示 XGBoost 特徵重要性圖表") and xgb_model is not None:
                    st.markdown("### 🎯 XGBoost 特徵重要性")
                    explainer = shap.Explainer(xgb_model.best_estimator_)
                    shap_values = explainer(X)
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
        
        # 綜合總結
        st.markdown("---")
        st.markdown("### 綜合總結：買定唔買")
        if score >= 5 or (up_proba is not None and up_proba > 0.6 and forecast_trend == "上漲"):
            reasons = ["技術指標強勢"]
            if any("頭肩底" in item or "雙底" in item for item in explanation):
                reasons.append("圖形形態支持反轉")
            if rf_proba is not None and rf_proba > 0.6:
                reasons.append("隨機森林預測看漲")
            if lgbm_proba is not None and lgbm_proba > 0.6:
                reasons.append("LightGBM 預測看漲")
            if arima_trend is not None and arima_trend == "上漲":
                reasons.append("ARIMA 預測看漲")
            st.success(f"✅ **建議：買** - {', '.join(reasons)}")
        else:
            reason = []
            if score < 3:
                reason.append(f"技術指標偏弱（評分 {score:.1f}）")
            if up_proba is None or up_proba < 0.6:
                reason.append(f"XGBoost 短期上升機率低（{up_proba*100:.2f}%）")
            if rf_proba is None or rf_proba < 0.6:
                reason.append(f"隨機森林短期上升機率低（{rf_proba*100:.2f}%）")
            if lgbm_proba is None or lgbm_proba < 0.6:
                reason.append(f"LightGBM 短期上升機率低（{lgbm_proba*100:.2f}%）")
            if forecast_trend != "上漲":
                reason.append("Prophet 預測 30 天趨勢不明")
            if arima_trend is None or arima_trend != "上漲":
                reason.append("ARIMA 預測 30 天趨勢不明")
            st.error(f"❌ **建議：唔買** - {', '.join(reason)}")
    else:
        st.error(f"❌ 無法獲取 {selected_symbol} 的數據，請檢查代碼或網絡連線。")
else:
    st.info("💡 請選擇股票或輸入代碼開始分析。")
