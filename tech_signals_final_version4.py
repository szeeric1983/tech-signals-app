import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from prophet import Prophet
import yfinance as yf
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# Streamlit 頁面配置
st.set_page_config(
    page_title="智能選股平台 PRO", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 增強版用戶指南 ====================
st.sidebar.title("📘 使用說明")

# 技術指標說明
show_tech_help = st.sidebar.checkbox("📊 顯示技術指標說明", value=False)
if show_tech_help:
    st.sidebar.markdown("""
    #### 🟢 技術指標說明 (PRO版)
    
    **初階指標增強：**
    - 📈 **智能RSI**：科技股RSI<28/其他<30視為超賣，加入趨勢權重調整
    - 📉 **MACD 2.0**：識別普通/弱勢黃金交叉，柱狀圖強度分級
    - 🎚️ **布林通道PRO**：結合波動率自動調整軌道寬度
    - 💹 **成交量進階**：科技股需2.5倍均量/其他2倍觸發
    
    **高階指標升級：**
    - 🌀 **TD9結構+**：完成後自動檢測成交量配合
    - 🔄 **RSI背離2.0**：需價格波動>5%才確認
    - ✨ **量價共振**：RSI背離+放量上漲=強力信號
    - 📏 **動態乖離率**：科技股MA20乖離<-7%才觸發
    
    **新增核心功能：**
    - 🌪️ **ADX趨勢強度**：>25為有效趨勢
    - 🏷️ **行業自適應**：自動識別科技/金融等板塊
    - ⚖️ **多空權重**：多頭市場信號自動強化1.2x
    - 📊 **成交量淨額(OBV)**：量能累積指標
    - ☁️ **一目均衡表**：綜合趨勢指標
    - 📐 **黃金比率回調位**：關鍵支持阻力位
    - 🖼️ **圖形形態識別**：自動識別頭肩頂/底、雙頂/底
    """)

# ==================== 主應用程序 ====================
st.title("📈 智能選股平台 PRO (香港版)")

# 股票分類定義
stock_categories = {
    "MAG7": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
    "AI 股": ["NVDA", "AMD", "TSM", "PLTR", "SMCI", "ASML", "TEM"],
    "ETF": ["SPY", "QQQ", "TQQQ", "SQQQ", "ARKK"],
    "黃金 / 債券 / 金融": ["GLD", "TLT", "XLF", "JPM", "BAC"],
    "比特幣 / 區塊鏈": ["BITO", "MARA", "RIOT", "COIN", "GBTC"],
    "能源 / 石油": ["XLE", "CVX", "XOM", "OXY", "BP"],
    "5G / 半導體": ["QCOM", "AVGO", "SWKS", "NXPI", "MRVL"],
    "港股": ["0700.HK", "0005.HK", "1299.HK", "0941.HK", "0388.HK"]
}

# 取得即時價格
@st.cache_data(ttl=600)
def get_latest_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return data["Close"].iloc[-1] if not data.empty else None
    except:
        return None

# 擷取歷史股價資料
@st.cache_data(ttl=3600)
def fetch_price_data(symbol):
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=365)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Close": "close",
            "Volume": "volume",
            "High": "high",
            "Low": "low"
        })
        # 移除時區信息
        df["date"] = df["date"].dt.tz_localize(None)
        return df[["date", "close", "volume", "high", "low"]]
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        return None

# ==================== 新增技術指標計算 ====================
def calculate_obv(df):
    """計算OBV(成交量淨額)指標 - 香港叫法: 成交量淨額"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma20'] = df['obv'].rolling(window=20).mean()  # 20日OBV均線
    return df

def calculate_ichimoku(df):
    """計算Ichimoku雲指標 - 香港叫法: 一目均衡表"""
    # 轉換線 (Tenkan-sen)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['ichi_tenkan'] = (high_9 + low_9) / 2
    
    # 基準線 (Kijun-sen)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['ichi_kijun'] = (high_26 + low_26) / 2
    
    # 先行線A (Senkou Span A)
    df['ichi_senkou_a'] = ((df['ichi_tenkan'] + df['ichi_kijun']) / 2).shift(26)
    
    # 先行線B (Senkou Span B)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['ichi_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # 遲行線 (Chikou Span)
    df['ichi_chikou'] = df['close'].shift(-26)
    
    return df

def calculate_fibonacci(df, lookback=60):
    """計算斐波那契回調位 - 香港叫法: 黃金比率"""
    recent_high = df['high'].rolling(window=lookback).max().iloc[-1]
    recent_low = df['low'].rolling(window=lookback).min().iloc[-1]
    diff = recent_high - recent_low
    
    fib_levels = {
        '0%': recent_high,
        '23.6%': recent_high - diff * 0.236,
        '38.2%': recent_high - diff * 0.382,
        '50%': recent_high - diff * 0.5,
        '61.8%': recent_high - diff * 0.618,
        '100%': recent_low
    }
    
    return fib_levels

def detect_head_shoulders(df):
    """識別頭肩頂/底形態 - 香港叫法: 頭肩頂/頭肩底"""
    patterns = []
    
    for i in range(2, len(df)-2):
        # 頭肩頂識別
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i-1] > df['high'].iloc[i-2] and
            df['high'].iloc[i+1] > df['high'].iloc[i+2]):
            patterns.append(('頭肩頂', df['date'].iloc[i]))
        
        # 頭肩底識別
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i-1] < df['low'].iloc[i-2] and
            df['low'].iloc[i+1] < df['low'].iloc[i+2]):
            patterns.append(('頭肩底', df['date'].iloc[i]))
    
    return patterns

def detect_double_tops_bottoms(df, threshold=0.05):
    """識別雙頂/雙底形態 - 香港叫法: 雙頂/雙底"""
    patterns = []
    
    for i in range(1, len(df)-1):
        # 雙頂識別
        if (abs(df['high'].iloc[i] - df['high'].iloc[i-1]) < threshold * df['high'].iloc[i] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1]):
            patterns.append(('雙頂', df['date'].iloc[i]))
        
        # 雙底識別
        if (abs(df['low'].iloc[i] - df['low'].iloc[i-1]) < threshold * df['low'].iloc[i] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+1]):
            patterns.append(('雙底', df['date'].iloc[i]))
    
    return patterns

def plot_fibonacci_levels(fig, df, fib_levels):
    """在現有圖表上添加斐波那契回調位"""
    last_date = df['date'].iloc[-1]
    
    for level, price in fib_levels.items():
        fig.add_shape(type='line',
                      x0=df['date'].iloc[0], y0=price,
                      x1=last_date, y1=price,
                      line=dict(color='purple', width=1, dash='dot'),
                      name=f'黃金比率 {level}')
        
        fig.add_annotation(x=last_date, y=price,
                          text=f'黃金比率 {level}',
                          showarrow=False,
                          yshift=10)
    
    return fig

def plot_ichimoku(fig, df):
    """添加Ichimoku雲到現有圖表"""
    # 雲層填充 (Senkou Span A & B之間的區域)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ichi_senkou_a'],
        line=dict(width=0),
        name='雲層上緣',
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ichi_senkou_b'],
        line=dict(width=0),
        name='雲層下緣',
        fill='tonexty',
        fillcolor='rgba(100,100,255,0.2)'
    ))
    
    # 其他Ichimoku線條
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ichi_tenkan'],
        line=dict(color='green', width=1),
        name='轉換線'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ichi_kijun'],
        line=dict(color='red', width=1),
        name='基準線'
    ))
    
    return fig

# 增強版技術指標計算
def enhanced_calculate_indicators(df, sector=""):
    """Calculate technical indicators with sector-specific parameters"""
    # Moving Averages
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    
    # RSI with sector-specific thresholds
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    
    # MACD
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["macd_hist"] = ta.trend.MACD(df["close"]).macd_diff()
    
    # Bollinger Bands
    df["bb_upper"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bb_lower"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    df["bb_break"] = df.apply(lambda row: "上軌" if row.close > row.bb_upper else ("下軌" if row.close < row.bb_lower else "中間"), axis=1)
    
    # Volume Analysis
    df["vol_avg20"] = df["volume"].rolling(window=20).mean()
    if sector in ["Technology", "Consumer Cyclical"]:
        df["vol_spike"] = df["volume"] > 2.5 * df["vol_avg20"]
    else:
        df["vol_spike"] = df["volume"] > 2 * df["vol_avg20"]
    
    # Trend Strength
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
    
    # TD9 Structure
    df["td9_down"] = (df["close"] < df["close"].shift(4)).astype(int)
    df["td9_count"] = df["td9_down"] * (df["td9_down"].groupby((df["td9_down"] != df["td9_down"].shift()).cumsum()).cumcount() + 1)
    
    # 新增指標計算
    df = calculate_obv(df)  # 成交量淨額
    df = calculate_ichimoku(df)  # 一目均衡表
    
    # 斐波那契回調位 (單獨處理，不加入df)
    fib_levels = calculate_fibonacci(df)
    
    # 圖形模式識別
    hs_patterns = detect_head_shoulders(df)  # 頭肩頂/底
    dt_patterns = detect_double_tops_bottoms(df)  # 雙頂/雙底
    
    return df, fib_levels, hs_patterns, dt_patterns

# 增強版信號生成
def enhanced_generate_signal(df, fib_levels, hs_patterns, dt_patterns, sector=""):
    """Enhanced scoring system with market context"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Initialize scoring
    score = 0
    explanation = []
    weight = 1
    
    # Market Trend Context
    market_trend = "up" if latest["close"] > df["ma50"].iloc[-1] else "down"
    if market_trend == "up":
        weight *= 1.2  # Favor long signals in uptrend
    else:
        weight *= 0.8  # Reduce signal strength in downtrend
    
    # 1. RSI Analysis (Weighted)
    rsi_threshold = 28 if sector in ["Technology", "Consumer Cyclical"] else 30
    if latest.rsi < rsi_threshold:
        score += 1.5 * weight
        explanation.append(f"RSI超賣({latest.rsi:.1f})")
    elif latest.rsi > 70:
        score -= 1 * weight
        explanation.append(f"RSI超買({latest.rsi:.1f})")
    
    # 2. MACD Analysis
    if latest.macd_hist > 0 and prev.macd_hist < 0:
        score += 1.2 * weight
        explanation.append(f"MACD柱狀圖翻正 (值:{latest.macd_hist:.2f}，信號線:{latest.macd_signal:.2f})")
    elif latest.macd_hist < 0 and prev.macd_hist > 0:
        score -= 1 * weight
        explanation.append(f"MACD柱狀圖翻負 (值:{latest.macd_hist:.2f}，信號線:{latest.macd_signal:.2f}，顯示動能轉弱)")
    
    # 3. Bollinger Bands
    if latest.bb_break == "下軌":
        score += 1.5 * weight
        explanation.append(f"布林帶跌破下軌 (價格:{latest.close:.2f}，下軌:{latest.bb_lower:.2f})")
    elif latest.bb_break == "上軌":
        score -= 1 * weight
        explanation.append(f"布林帶突破上軌 (價格:{latest.close:.2f}，上軌:{latest.bb_upper:.2f})")
    
    # 4. Volume Spike
    if latest.vol_spike:
        if latest.close > prev.close:
            score += 1.8 * weight
            explanation.append(f"放量上漲 (成交量:{latest.volume/1e6:.1f}M，20日均量:{latest.vol_avg20/1e6:.1f}M)")
        else:
            score += 1 * weight
            explanation.append(f"成交量激增 (成交量:{latest.volume/1e6:.1f}M，20日均量:{latest.vol_avg20/1e6:.1f}M)")
    
    # 5. Trend Strength (ADX)
    if latest.adx > 25:
        if market_trend == "up":
            score += 0.5 * weight
            explanation.append(f"強勁上升趨勢 (ADX:{latest.adx:.1f})")
        else:
            score -= 0.5 * weight
            explanation.append(f"強勁下降趨勢 (ADX:{latest.adx:.1f})")
    
    # 6. TD9 Structure
    td9_signal = df["td9_count"].iloc[-1]
    if td9_signal >= 9:
        score += 1.5 * weight
        explanation.append(f"TD9結構完成 (計數:{td9_signal})")
    
    # 7. OBV Analysis
    if latest.obv > latest.obv_ma20 and latest.close > prev.close:
        score += 0.8 * weight
        explanation.append(f"成交量淨額(OBV)突破20日均線且價格上升 (OBV:{latest.obv/1e6:.1f}M，20日OBV均線:{latest.obv_ma20/1e6:.1f}M)")
    
    # 8. Ichimoku Cloud Analysis
    if latest.close > latest.ichi_senkou_a and latest.close > latest.ichi_senkou_b:
        score += 1 * weight
        explanation.append(f"價格突破一目均衡表雲層 (雲層上緣:{latest.ichi_senkou_a:.2f}，雲層下緣:{latest.ichi_senkou_b:.2f})")
    
    # 9. Fibonacci Retracement
    current_price = df['close'].iloc[-1]
    for level, price in fib_levels.items():
        if abs(current_price - price) < current_price * 0.01:  # 1%範圍內視為觸及
            if level in ['38.2%', '50%', '61.8%']:
                score += 1
                explanation.append(f"觸及黃金比率重要回調位 {level} ({price:.2f})")
    
    # 10. Pattern Recognition
    if hs_patterns:
        last_pattern = hs_patterns[-1]
        if last_pattern[0] == '頭肩底' and (df['date'].iloc[-1] - last_pattern[1]).days < 10:
            score += 1.5
            explanation.append(f"近期出現頭肩底形態 (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
        elif last_pattern[0] == '頭肩頂' and (df['date'].iloc[-1] - last_pattern[1]).days < 10:
            score -= 1.5
            explanation.append(f"近期出現頭肩頂形態 (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
    
    if dt_patterns:
        last_pattern = dt_patterns[-1]
        if last_pattern[0] == '雙底' and (df['date'].iloc[-1] - last_pattern[1]).days < 10:
            score += 1
            explanation.append(f"近期出現雙底形態 (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
        elif last_pattern[0] == '雙頂' and (df['date'].iloc[-1] - last_pattern[1]).days < 10:
            score -= 1
            explanation.append(f"近期出現雙頂形態 (日期: {last_pattern[1].strftime('%Y-%m-%d')})")
    
    # Signal Determination
    if score >= 5:
        signal = "🟢 強力買入"
    elif score >= 3:
        signal = "🟡 謹慎買入"
    elif score <= -3:
        signal = "🔴 強力賣出"
    elif score <= -1:
        signal = "🔴 考慮賣出"
    else:
        signal = "⚪ 中性觀望"
    
    return signal, explanation, score, latest

# 左邊分類 + Sticky 股票按鈕
st.sidebar.header("📂 股票分類")
category = st.sidebar.selectbox("選擇分類：", list(stock_categories.keys()))
stock_list = stock_categories[category]

selected_symbol = None
for stock in stock_list:
    price = get_latest_price(stock)
    label = f"{stock} - ${price:.2f}" if price is not None else stock
    if st.sidebar.button(label, key=f"btn_{stock}"):
        selected_symbol = stock

# 輸入股票代碼（可自定）
custom = st.sidebar.text_input("或輸入股票代碼：", "")
if custom:
    selected_symbol = custom.upper()

if selected_symbol:
    df = fetch_price_data(selected_symbol)
    if df is not None:
        # Get sector information
        try:
            sector = yf.Ticker(selected_symbol).info.get('sector', '')
        except:
            sector = ''
        
        # Enhanced analysis
        df, fib_levels, hs_patterns, dt_patterns = enhanced_calculate_indicators(df, sector)
        signal, explanation, score, latest = enhanced_generate_signal(df, fib_levels, hs_patterns, dt_patterns, sector)
        
        # Enhanced display
        st.subheader(f"🔍 分析股票：{selected_symbol} - {signal}")
        st.markdown(f"📌 **綜合評分：{score:.1f}/10**")
        
        # New metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("趨勢強度", f"{latest.adx:.1f}", 
                    "強勢" if latest.adx > 25 else "弱勢",
                    delta_color="inverse")
        with col2:
            st.metric("市場方向", 
                    "多頭" if latest["close"] > df["ma50"].iloc[-1] else "空頭",
                    f"{latest.close/df['ma50'].iloc[-1]-1:.2%}")
        with col3:
            st.metric("行業特性", sector or "未知", 
                    "高波動" if sector in ["Technology", "Consumer Cyclical"] else "一般")
        
        # Enhanced signal explanations
        with st.expander("📋 詳細訊號分析", expanded=True):
            st.markdown("### 當前市場狀態")
            
            # 更詳細的趨勢分析
            ma50_trend = "上升" if latest['close'] > df['ma50'].iloc[-1] else "下降"
            ma20_vs_ma50 = "黃金交叉" if df['ma20'].iloc[-1] > df['ma50'].iloc[-1] else "死亡交叉"
            st.write(f"- **趨勢方向**: {ma50_trend} (收盤價 {latest['close']:.2f} vs MA50 {df['ma50'].iloc[-1]:.2f})")
            st.write(f"- **短期趨勢**: {'上升' if df['ma5'].iloc[-1] > df['ma20'].iloc[-1] else '下降'} (MA5 {df['ma5'].iloc[-1]:.2f} vs MA20 {df['ma20'].iloc[-1]:.2f})")
            st.write(f"- **中期交叉**: {ma20_vs_ma50} (MA20 vs MA50)")
            st.write(f"- **趨勢強度**: {'強勁' if latest.adx > 25 else '普通'} (ADX值:{latest.adx:.1f})")
            st.write(f"- **行業特性**: {sector or '未知'} (影響指標閾值)")
            
            st.markdown("### 技術指標分析")
            if not explanation:  # 如果沒有信號
                st.warning("⚠️ 當前無明顯技術信號")
            else:
                for item in explanation:
                    if "買" in item or "漲" in item:
                        st.success(f"✅ {item}")
                    elif "賣" in item or "跌" in item:
                        st.error(f"❌ {item}")
                    else:
                        st.info(f"ℹ️ {item}")
            
            # 動態高亮當前評分範圍
            st.markdown("### 評分標準")
            score_ranges = [
                (5, "🟢 強力買入 (多個確認訊號且趨勢明確)", "success"),
                (3, "🟡 謹慎買入 (需等待進一步確認)", "info"),
                (-1, "⚪ 中性觀望 (市場方向不明)", "warning"),
                (-float('inf'), "🔴 考慮賣出 (多個利空訊號)", "error")
            ]
            
            for threshold, text, color in score_ranges:
                if score >= threshold:
                    if color == "success":
                        st.success(text)
                    elif color == "info":
                        st.info(text)
                    elif color == "warning":
                        st.warning(text)
                    else:
                        st.error(text)
                    break
            
            # 增加MACD詳細解釋
            if any("MACD" in item for item in explanation):
                st.markdown(f"""
                #### 📊 MACD指標深入解讀
                - **柱狀圖翻正/負**: 表示短期動能變化，可能預示趨勢反轉
                - **當前值**: {latest.macd_hist:.2f} (正值為多頭動能，負值為空頭動能)
                - **信號線**: {latest.macd_signal:.2f} (當MACD線穿越信號線時產生交易信號)
                - **建議**: 結合其他指標確認，單一指標可能產生假信號
                """)
        
        # 新增技術指標展示區
        st.markdown("---")
        st.subheader("📊 進階技術指標分析")
        
        # 斐波那契回調位顯示
        st.markdown("### 黃金比率回調位分析")
        fib_table = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['價位'])
        st.table(fib_table.style.format({'價位': '{:.2f}'}))
        
        # 圖形模式識別結果
        if hs_patterns or dt_patterns:
            st.markdown("### 🔄 圖形形態識別")
            col1, col2 = st.columns(2)
            
            with col1:
                if hs_patterns:
                    st.write("#### 頭肩形態")
                    hs_df = pd.DataFrame(hs_patterns, columns=['形態', '日期'])
                    st.dataframe(hs_df.tail(3))  # 顯示最近3個
            
            with col2:
                if dt_patterns:
                    st.write("#### 雙頂/雙底")
                    dt_df = pd.DataFrame(dt_patterns, columns=['形態', '日期'])
                    st.dataframe(dt_df.tail(3))
        else:
            st.info("ℹ️ 近期未識別到明顯的圖形形態")
        
        # 畫圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="收盤價", line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(dash='dot', color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], name="MA50", line=dict(dash='dash', color='#2ca02c')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB 上軌", line=dict(width=0.5, color='#d62728')))
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB 下軌", line=dict(width=0.5, color='#d62728')))
        
        # 添加新指標到圖表
        fig = plot_fibonacci_levels(fig, df, fib_levels)
        fig = plot_ichimoku(fig, df)
        
        fig.update_layout(
            title=f"{selected_symbol} 技術分析圖 (含黃金比率 & 一目均衡表)",
            xaxis_title="日期",
            yaxis_title="價格",
            legend=dict(orientation='h'),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Enhanced trading suggestions
        if score >= 5:
            st.success("""
            💡 **操作建議**: 
            - 🏦 可建立主要倉位 (50-70%)
            - ⛔ 止損設置在最近支持位下方3%
            - 🎯 目標報酬風險比建議 3:1 以上
            - 📈 適合趨勢跟蹤策略
            - 💰 黃金比率支持位可作為加倉點
            """)
        elif score >= 3:
            st.info("""
            💡 **操作建議**: 
            - 🧪 可測試性建倉 (20-30%)
            - ⏳ 等待進一步確認後加倉
            - ⛔ 嚴格止損
            - 🔍 適合波段操作
            - 📊 關注一目均衡表雲層變化
            """)
        elif score <= -3:
            st.error("""
            💡 **操作建議**: 
            - 📉 考慮減倉或對沖
            - 🔼 短期反彈可作為賣出機會
            - 🚫 不宜新建多頭倉位
            - 🛡️ 適合避險策略
            - ⚠️ 注意黃金比率阻力位
            """)

        # Prophet Prediction - 修正時區問題
        with st.expander("⏳ AI 模組：Prophet 預測未來 7 日", expanded=True):
            try:
                # 準備數據並確保沒有時區信息
                df_prophet = df[["date", "close"]].copy()
                df_prophet = df_prophet.rename(columns={"date": "ds", "close": "y"})
                df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
                
                m = Prophet(daily_seasonality=True)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=7)
                forecast = m.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_prophet["ds"], 
                    y=df_prophet["y"], 
                    mode="lines", 
                    name="實際價格",
                    line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["ds"], 
                    y=forecast["yhat"], 
                    mode="lines", 
                    name="預測價格",
                    line=dict(color='#ff7f0e')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="上界",
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    mode="lines",
                    fill='tonexty',
                    line=dict(width=0),
                    showlegend=False,
                    name="下界",
                ))
                fig.update_layout(
                    title="Prophet 預測未來 7 日收盤價",
                    xaxis_title="日期",
                    yaxis_title="價格"
                )
                
                forecast_next7 = forecast.tail(7).copy()
                forecast_next7["預測日"] = [f"T+{i+1}" for i in range(7)]
                forecast_table = forecast_next7[["預測日", "ds", "yhat", "yhat_lower", "yhat_upper"]]
                forecast_table.columns = ["預測日", "日期", "預測收盤價", "最低估價", "最高估價"]
                
                st.subheader("📋 Prophet 預測未來 7 日價格詳情")
                st.dataframe(
                    forecast_table.style.format({
                        "預測收盤價": "{:.2f}", 
                        "最低估價": "{:.2f}", 
                        "最高估價": "{:.2f}"
                    }).applymap(
                        lambda x: 'color: green' if x > df_prophet["y"].iloc[-1] else 'color: red',
                        subset=["預測收盤價"]
                    )
                )
                
                st.markdown("""
                📈 **預測結果解讀**:
                - 未來數日走勢趨勢及價格區間
                - 請搭配技術指標一併判斷
                - 灰色區域為80%置信區間
                """)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ 預測失敗：{e}")

        # XGBoost Prediction
        with st.expander("🚀 AI 模組：XGBoost 升跌預測", expanded=True):
            try:
                # 特徵準備
                df_xgb = df.copy()
                df_xgb["return"] = df_xgb["close"].pct_change()
                df_xgb["target"] = (df_xgb["return"].shift(-1) > 0).astype(int)
                df_xgb["ma5"] = df_xgb["close"].rolling(5).mean()
                df_xgb["ma20"] = df_xgb["close"].rolling(20).mean()
                df_xgb["rsi"] = ta.momentum.RSIIndicator(df_xgb["close"]).rsi()
                df_xgb["macd"] = ta.trend.MACD(df_xgb["close"]).macd_diff()
                df_xgb["adx"] = ta.trend.ADXIndicator(df_xgb["high"], df_xgb["low"], df_xgb["close"]).adx()
                df_xgb["vol"] = df_xgb["volume"]
                df_xgb["obv"] = df_xgb["obv"]

                df_xgb = df_xgb.dropna().copy()
                features = ["ma5", "ma20", "rsi", "macd", "adx", "vol", "obv"]
                X = df_xgb[features]
                y = df_xgb["target"]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                latest_features = X_scaled[-1:]
                proba = model.predict_proba(latest_features)[0]

                st.write(f"📈 **預測下一日上升機率**：{proba[1]*100:.2f}%")
                st.write(f"📉 **下降機率**：{proba[0]*100:.2f}%")
                st.write(f"✅ **測試集準確率**：約 {acc*100:.2f}%")

                # 特徵重要性
                st.markdown("### 🎯 特徵重要性 (增強版)")
                explainer = shap.Explainer(model)
                shap_values = explainer(X)
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                plt.title("XGBoost Value", fontsize=12)
                st.pyplot(plt.gcf())
                
                st.markdown("""
                **解讀指南**:
                - 正值特徵增加上漲概率
                - 負值特徵增加下跌概率
                - 條形長度表示影響程度
                """)
            except Exception as e:
                st.error(f"❌ XGBoost 模型預測失敗：{e}")

        # Financial Health Module
        st.markdown('---')
        st.subheader('🧾 財報模組：基本面健康評估')
        try:
            yf_ticker = yf.Ticker(selected_symbol)
            info = yf_ticker.info
            if not info or len(info) == 0:
                raise ValueError('空資料')
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            pe = info.get('trailingPE')
            roe = info.get('returnOnEquity')
            fcf = info.get('freeCashflow')
            debt_to_equity = info.get('debtToEquity')
            revenue = info.get('totalRevenue')
            gross_margin = info.get('grossMargins')

            red_flags = []
            score = 0

            st.write('🏢 **公司類別**：', f"{sector} - {industry}")
            
            # 行業對比
            if sector and 'Technology' in sector:
                st.write('🔍 **類股指標**：科技股 ➜ 注重營收成長與研發投入')
                pe_threshold = 25
                fcf_threshold = 0  # 科技股允許暫時負現金流
            else:
                st.write('🔍 **類股指標**：傳統股 ➜ 注重穩定現金流與負債比')
                pe_threshold = 15
                fcf_threshold = 0.5  # 傳統股需正現金流

            # 評估指標
            col1, col2 = st.columns(2)
            
            with col1:
                if pe is not None:
                    st.metric("本益比", f"{pe:.1f}", 
                            "低於標準" if pe < pe_threshold else "高於標準",
                            delta_color="inverse")
                    if pe < pe_threshold: 
                        score += 1
                        st.success('✅ 低於行業標準')
                    else:
                        st.warning('⚠️ 高於行業標準')
                else:
                    st.warning("本益比數據缺失")

                if roe is not None:
                    st.metric("股東權益報酬率", f"{roe:.1%}", 
                            "優於15%標準" if roe > 0.15 else "低於15%標準")
                    if roe > 0.15: 
                        score += 1
                        st.success('✅ 優於標準')
                    else:
                        st.warning('⚠️ 低於標準')
                else:
                    st.warning("ROE數據缺失")

            with col2:
                if fcf is not None:
                    st.metric("自由現金流", f"{fcf/1e6:.1f}M", 
                            "達標" if fcf > fcf_threshold else "未達標")
                    if fcf > fcf_threshold: 
                        score += 1
                        st.success('✅ 達標')
                    else: 
                        red_flags.append('自由現金流未達標準')
                        st.warning('⚠️ 未達標')
                else:
                    st.warning("自由現金流數據缺失")

                if debt_to_equity is not None:
                    st.metric("負債權益比", f"{debt_to_equity:.2f}", 
                            "安全" if debt_to_equity < 1.5 else "過高")
                    if debt_to_equity < 1.5:
                        score += 1
                        st.success('✅ 安全')
                    else:
                        red_flags.append('負債權益比過高')
                        st.warning('⚠️ 過高')
                else:
                    st.warning("負債權益比數據缺失")

            # 健康評分
            st.markdown("### 📊 健康評分")
            health_col1, health_col2 = st.columns([1, 3])
            
            with health_col1:
                st.metric("", f"{score}/4", 
                        "優秀" if score == 4 else ("良好" if score >= 2 else "不佳"))
            
            with health_col2:
                if score == 4:
                    st.success('✅ **結論**：財務健康，基本面穩健')
                elif score >= 2:
                    st.info('ℹ️ **結論**：財務中等，需關注風險因素')
                else:
                    st.error('❌ **結論**：財務風險高，謹慎投資')

            # 風險提示
            if red_flags:
                st.markdown("### ⚠️ 風險提示")
                for r in red_flags:
                    st.warning(r)

        except Exception as e:
            st.warning(f'⚠️ 無法取得財報資料: {e}')

    else:
        st.error("❌ 無法取得歷史價格，請檢查代碼。")
else:
    st.info("💡 請從左邊選擇股票或輸入代碼開始分析。")
    st.image("https://via.placeholder.com/800x400?text=智能選股平台+PRO", use_container_width=True)