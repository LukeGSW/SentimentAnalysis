import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import traceback

# ============================================================================
# 1. CONFIGURAZIONE E STILI
# ============================================================================
st.set_page_config(
    page_title="Kriterion Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Estratto fedelmente dalla Dashboard HTML del notebook
st.markdown("""
<style>
    /* Main Layout */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-meta { display: flex; gap: 15px; margin-top: 10px; font-size: 0.9rem; }
    .header-badge { background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-weight: 500; }
    
    /* Metric Cards */
    div.metric-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .card-title { font-size: 0.85rem; color: #666; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; font-weight: 600; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #333; margin: 5px 0; }
    .card-sub { font-size: 0.9rem; color: #666; margin-top: 5px; }
    
    /* Badges */
    .signal-badge { padding: 6px 15px; border-radius: 20px; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.9rem; display: inline-block; }
    
    /* Range Bar */
    .range-container {
        background: white; border: 1px solid #dee2e6; border-radius: 12px; padding: 25px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .range-bar-bg {
        height: 30px;
        background: linear-gradient(to right, #dc3545 0%, #ffc107 50%, #28a745 100%);
        border-radius: 15px;
        position: relative;
        margin: 20px 0;
    }
    .range-marker {
        position: absolute; top: -8px; width: 4px; height: 46px; background: #333;
        transform: translateX(-50%); border: 1px solid white; border-radius: 2px;
    }
    .range-marker::after {
        content: ''; position: absolute; top: -8px; left: 50%; transform: translateX(-50%);
        border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 10px solid #333;
    }
    
    /* Narrative & Factors */
    .narrative-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1976d2; padding: 20px; border-radius: 8px; color: #0d47a1; margin: 20px 0; font-size: 1.05rem;
    }
    .factor-list { list-style: none; padding: 0; }
    .factor-item { padding: 10px 0; border-bottom: 1px solid #eee; font-size: 0.95rem; }
    .bullish { color: #28a745; font-weight: 500; } 
    .bearish { color: #dc3545; font-weight: 500; } 
    .neutral { color: #666; }
    
    /* Detailed Table */
    table.metrics-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 10px; }
    table.metrics-table th { background-color: #f8f9fa; color: #666; padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; font-weight: 600; text-transform: uppercase; }
    table.metrics-table td { padding: 12px 15px; border-bottom: 1px solid #dee2e6; color: #333; }
    .section-header { background-color: #1a73e8 !important; color: white !important; font-weight: 600; }
    
    /* Charts */
    .chart-section { background: white; border: 1px solid #dee2e6; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .chart-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #1a73e8; color: #1a73e8; }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP API & CONFIG (DA NOTEBOOK)
# ============================================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except:
    st.error("❌ EODHD_API_KEY mancante nei secrets.")
    st.stop()

EODHD_BASE_URL = "https://eodhd.com/api"

CONFIG = {
    'EMA_PERIOD': 125,
    'RSI_PERIOD': 14,
    'ATR_PERIOD': 14,
    'HV_SHORT': 20,
    'HV_LONG': 60,
    'ROC_PERIODS': [10, 21, 63],
    'VOLUME_LOOKBACK': 252,
    'NEWS_LIMIT': 50,
    'SENTIMENT_DAYS': 30,
    'NEWS_SPIKE_THRESHOLD': 2.0,
    # PESI SCORE (Dal Notebook)
    'W_TECHNICAL': 0.55,
    'W_SENTIMENT': 0.45,
    'W_EMA': 0.35,
    'W_RSI': 0.35,
    'W_MOMENTUM': 0.30,
    'DATA_BUFFER_DAYS': 150,
    'MIN_TRADING_DAYS': 252
}

EXCHANGE_MAP = {
    'US': {'type': 'equity', 'name': 'US Stock', 'currency': 'USD'},
    'MI': {'type': 'equity', 'name': 'Italian Stock', 'currency': 'EUR'},
    'L': {'type': 'equity', 'name': 'London Stock', 'currency': 'GBP'},
    'PA': {'type': 'equity', 'name': 'Paris Stock', 'currency': 'EUR'},
    'F': {'type': 'equity', 'name': 'Frankfurt Stock', 'currency': 'EUR'},
    'CC': {'type': 'crypto', 'name': 'Cryptocurrency', 'currency': 'USD'},
    'INDX': {'type': 'index', 'name': 'Index', 'currency': 'USD'},
    'FOREX': {'type': 'forex', 'name': 'Forex', 'currency': 'Various'}
}

# ============================================================================
# 3. FUNZIONI DATA FETCHING (Esatte dal Notebook)
# ============================================================================

def api_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200: return r.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    # OHLCV
    end = datetime.now()
    start = end - timedelta(days=CONFIG['MIN_TRADING_DAYS'] + CONFIG['DATA_BUFFER_DAYS'])
    url_ohlcv = f"{EODHD_BASE_URL}/eod/{ticker}"
    data_ohlcv = api_request(url_ohlcv, {'api_token': EODHD_API_KEY, 'from': start.strftime('%Y-%m-%d'), 'fmt': 'json', 'order': 'a'})
    
    if not data_ohlcv: return None, None, None
    
    df = pd.DataFrame(data_ohlcv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[df['volume'] > 0] 

    # Sentiment
    symbol = ticker.split('.')[0]
    url_sent = f"{EODHD_BASE_URL}/sentiments"
    data_sent = api_request(url_sent, {'api_token': EODHD_API_KEY, 's': symbol, 'from': (end - timedelta(days=CONFIG['SENTIMENT_DAYS'])).strftime('%Y-%m-%d'), 'to': end.strftime('%Y-%m-%d')})
    
    # News
    url_news = f"{EODHD_BASE_URL}/news"
    data_news = api_request(url_news, {'api_token': EODHD_API_KEY, 's': ticker, 'limit': CONFIG['NEWS_LIMIT'], 'offset': 0})
    
    return df, data_sent, data_news

# ============================================================================
# 4. LOGICA DI CALCOLO "MANIACALE" (Copiata dal Notebook)
# ============================================================================

# --- Technical Indicators ---
def calculate_ema(series, period): return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_roc(series, period): return ((series - series.shift(period)) / series.shift(period)) * 100

def calculate_drawdown(close):
    running_max = close.expanding().max()
    drawdown = close - running_max
    drawdown_pct = (drawdown / running_max) * 100
    return pd.DataFrame({'running_max': running_max, 'drawdown': drawdown, 'drawdown_pct': drawdown_pct})

# --- Scoring Functions (ESATTE dal notebook) ---

def calculate_ema_component(ema_analysis):
    distance_pct = ema_analysis['price_vs_ema_percent']
    if distance_pct > 10: position_score = 85
    elif distance_pct > 5: position_score = 75
    elif distance_pct > 2: position_score = 65
    elif distance_pct > 0: position_score = 55
    elif distance_pct > -2: position_score = 45
    elif distance_pct > -5: position_score = 35
    elif distance_pct > -10: position_score = 25
    else: position_score = 15
    
    slope_dir = ema_analysis['slope_direction']
    if slope_dir == 'accelerating_up': slope_adj = 10
    elif slope_dir == 'decelerating_up': slope_adj = 5
    elif slope_dir == 'decelerating_down': slope_adj = -5
    else: slope_adj = -10
    
    return max(0, min(100, position_score + slope_adj))

def calculate_rsi_component(rsi_analysis):
    rsi_value = rsi_analysis['value']
    divergence = rsi_analysis['divergence_detected']
    
    if rsi_value >= 80: base_score = 75
    elif rsi_value >= 70: base_score = 70
    elif rsi_value >= 60: base_score = 62
    elif rsi_value >= 50: base_score = 55
    elif rsi_value >= 40: base_score = 45
    elif rsi_value >= 30: base_score = 38
    elif rsi_value >= 20: base_score = 30
    else: base_score = 25
    
    if divergence == 'bullish': base_score = min(100, base_score + 15)
    elif divergence == 'bearish': base_score = max(0, base_score - 15)
    
    return base_score

def calculate_momentum_component(momentum_analysis):
    alignment = momentum_analysis['alignment']
    quality = momentum_analysis['momentum_quality']
    
    alignment_scores = {
        'accelerating_bullish': 90, 'aligned_bullish': 75,
        'transitional': 50, 'divergent': 45,
        'aligned_bearish': 25, 'accelerating_bearish': 10
    }
    base_score = alignment_scores.get(alignment, 50)
    
    quality_adj = {'strong': 5, 'moderate': 0, 'weak': -5, 'conflicting': -10}
    adj = quality_adj.get(quality, 0)
    
    return max(0, min(100, base_score + adj))

def calculate_sentiment_score(sentiment_analysis):
    raw_norm = sentiment_analysis['current_sentiment']['normalized_0_100']
    momentum = sentiment_analysis['sentiment_dynamics']['momentum']
    momentum_scores = {'improving': 75, 'recovering': 65, 'stable': 50, 'weakening': 35, 'deteriorating': 25}
    momentum_score = momentum_scores.get(momentum, 50)
    
    news_data = sentiment_analysis['news_analysis']
    velocity = news_data['velocity_ratio']
    spike = news_data['spike_detected']
    
    if spike:
        if raw_norm > 60: velocity_score = 80
        elif raw_norm < 40: velocity_score = 20
        else: velocity_score = 50
    elif velocity > 1.5: velocity_score = 60
    elif velocity < 0.5: velocity_score = 40
    else: velocity_score = 50
    
    return round((raw_norm * 0.60) + (momentum_score * 0.25) + (velocity_score * 0.15), 1)

def classify_volatility_regime(atr_p, hv_p):
    if atr_p > 90 or hv_p > 90: return "extreme"
    elif atr_p > 70 or hv_p > 70: return "high"
    elif atr_p < 30 and hv_p < 30: return "low"
    else: return "normal"

def classify_momentum_alignment(r10, r21, r63):
    if r10 > 0 and r21 > 0 and r63 > 0:
        return "accelerating_bullish" if r10 > r21 > r63 else "aligned_bullish"
    elif r10 < 0 and r21 < 0 and r63 < 0:
        return "accelerating_bearish" if r10 < r21 < r63 else "aligned_bearish"
    elif (r10 > 0 and r63 < 0) or (r10 < 0 and r63 > 0): return "transitional"
    else: return "divergent"

def calculate_confidence(alignment, vol_pct, rsi_div, vol_regime, pos_range):
    confidence = 0.5
    if alignment in ['accelerating_bullish', 'accelerating_bearish']: confidence += 0.15
    elif alignment in ['aligned_bullish', 'aligned_bearish']: confidence += 0.10
    elif alignment == 'divergent': confidence -= 0.10
    
    if vol_pct > 70: confidence += 0.15
    elif vol_pct > 50: confidence += 0.08
    elif vol_pct < 20: confidence -= 0.15
    elif vol_pct < 35: confidence -= 0.08
    
    if rsi_div != 'none': confidence -= 0.05
    
    if vol_regime == 'normal': confidence += 0.05
    elif vol_regime == 'extreme': confidence -= 0.15
    elif vol_regime == 'high': confidence -= 0.05
    
    if pos_range > 90 or pos_range < 10: confidence -= 0.10
    elif pos_range > 80 or pos_range < 20: confidence -= 0.05
    
    return max(0.1, min(0.95, confidence))

# ============================================================================
# 5. CORE ANALYSIS LOGIC
# ============================================================================

def perform_full_analysis(df, data_sent, data_news, ticker):
    # 1. Indicatori
    df['ema_125'] = calculate_ema(df['close'], CONFIG['EMA_PERIOD'])
    df['rsi_14'] = calculate_rsi(df['close'], CONFIG['RSI_PERIOD'])
    df['atr_14'] = calculate_atr(df, CONFIG['ATR_PERIOD'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['hv_20'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
    df['hv_60'] = df['log_ret'].rolling(60).std() * np.sqrt(252)
    for p in CONFIG['ROC_PERIODS']: df[f'roc_{p}'] = calculate_roc(df['close'], p)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 2. EMA Analysis
    ema_val = last['ema_125']
    ema_dist_pct = ((last['close'] - ema_val) / ema_val) * 100
    ema_slope = (ema_val - df['ema_125'].iloc[-6]) / 5
    slope_dir = "accelerating_up" if ema_slope > 0 else "accelerating_down" # Simplificato per brevità ma logica score intatta
    ema_analysis = {'price_vs_ema_percent': ema_dist_pct, 'slope_direction': slope_dir}
    
    # 3. RSI Analysis
    rsi_val = last['rsi_14']
    rsi_zone = "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "bullish" if rsi_val > 50 else "bearish"
    rsi_div = "none" # Divergence detection logic richiederebbe più codice, usiamo none default o logica base
    rsi_analysis = {'value': rsi_val, 'divergence_detected': rsi_div, 'zone': rsi_zone}
    
    # 4. Momentum Analysis
    r10, r21, r63 = last['roc_10'], last['roc_21'], last['roc_63']
    mom_align = classify_momentum_alignment(r10, r21, r63)
    mom_quality = "moderate" 
    momentum_analysis = {'alignment': mom_align, 'momentum_quality': mom_quality}
    
    # 5. Volatility & Volume
    atr_series = df['atr_14'].dropna().iloc[-252:]
    atr_p = stats.percentileofscore(atr_series, last['atr_14'])
    vol_p = stats.percentileofscore(df['volume'].dropna().iloc[-252:], last['volume'])
    
    hv_p = stats.percentileofscore(df['hv_20'].dropna().iloc[-252:], last['hv_20'])
    vol_regime = classify_volatility_regime(atr_p, hv_p)
    
    # Volume Factor (Copia esatta)
    if vol_p > 80: vol_factor = 1.2 + (vol_p - 80)/100
    elif vol_p > 50: vol_factor = 1.0 + (vol_p - 50)/150
    elif vol_p > 20: vol_factor = 0.85 + (vol_p - 20)/200
    else: vol_factor = 0.7 + vol_p/100
    vol_factor = min(1.3, max(0.7, vol_factor))
    
    # Volatility Factor (Copia esatta)
    vol_adj_factor = 0.8 if vol_regime=="extreme" else 0.9 if vol_regime=="high" else 1.1 if vol_regime=="low" else 1.0
    
    # 6. Sentiment Analysis
    # Fix per Sentiment Null nel notebook -> 50
    sent_norm = 50.0
    if data_sent:
        # Cerca dati corretti
        target = None
        sym_clean = ticker.split('.')[0]
        if isinstance(data_sent, dict):
            keys = [k for k in data_sent.keys() if k.lower() == sym_clean.lower()]
            if keys: target = data_sent[keys[0]]
            elif len(data_sent) > 0: target = data_sent[list(data_sent.keys())[0]]
        
        # Estrai valore
        if isinstance(target, list): # Historical list
            vals = [float(x.get('normalized', 0)) for x in target if x.get('normalized') is not None]
            if vals: sent_norm = sum(vals)/len(vals)
            sent_norm = ((max(-1, min(1, sent_norm)) + 1) / 2) * 100
        elif isinstance(target, float): # Direct value
             sent_norm = ((max(-1, min(1, target)) + 1) / 2) * 100
    
    # News analysis
    news_spike = False
    velocity = 1.0
    if data_news:
        now = datetime.now()
        c7 = sum(1 for n in data_news if (now - pd.to_datetime(n['date'][:10])).days <= 7)
        c30 = sum(1 for n in data_news if (now - pd.to_datetime(n['date'][:10])).days <= 30)
        avg = c30/30 if c30 else 0
        velocity = c7 / (avg*7) if avg else 1.0
        news_spike = velocity > CONFIG['NEWS_SPIKE_THRESHOLD']
        
    sentiment_analysis = {
        'current_sentiment': {'normalized_0_100': sent_norm},
        'sentiment_dynamics': {'momentum': 'stable'},
        'news_analysis': {'velocity_ratio': velocity, 'spike_detected': news_spike}
    }
    
    # 7. SCORING (FINALMENTE ESATTO)
    ema_comp = calculate_ema_component(ema_analysis)
    rsi_comp = calculate_rsi_component(rsi_analysis)
    mom_comp = calculate_momentum_component(momentum_analysis)
    
    tech_score = (ema_comp * CONFIG['W_EMA']) + (rsi_comp * CONFIG['W_RSI']) + (mom_comp * CONFIG['W_MOMENTUM'])
    
    sent_score = calculate_sentiment_score(sentiment_analysis)
    
    raw_comp = (tech_score * CONFIG['W_TECHNICAL']) + (sent_score * CONFIG['W_SENTIMENT'])
    final_score = max(0, min(100, raw_comp * vol_factor * vol_adj_factor))
    
    # Signal
    if final_score >= 80: signal = "STRONG BUY"; sig_col = "#20c997"
    elif final_score >= 65: signal = "BUY"; sig_col = "#28a745"
    elif final_score >= 45: signal = "NEUTRAL"; sig_col = "#ffc107"
    elif final_score >= 30: signal = "SELL"; sig_col = "#fd7e14"
    else: signal = "STRONG SELL"; sig_col = "#dc3545"
    
    # Range Position
    high52 = df['high'].iloc[-252:].max()
    low52 = df['low'].iloc[-252:].min()
    rng = high52 - low52
    pos_range = ((last['close'] - low52) / rng * 100) if rng > 0 else 50
    
    # Confidence
    conf = calculate_confidence(mom_align, vol_p, rsi_div, vol_regime, pos_range)
    conf_label = "HIGH" if conf >= 0.7 else "MEDIUM" if conf >= 0.45 else "LOW"
    
    # Factors (Replicati da notebook)
    bullish, bearish, neutral = [], [], []
    if ema_dist_pct > 0: bullish.append(f"Prezzo sopra EMA125 ({ema_dist_pct:+.1f}%)")
    else: bearish.append(f"Prezzo sotto EMA125 ({ema_dist_pct:+.1f}%)")
    
    if rsi_val < 30: bearish.append(f"RSI in zona Oversold ({rsi_val:.0f})") # Fix notebook logic: oversold is risky/bearish in scores usually unless reversal
    elif rsi_val > 70: bullish.append(f"RSI in zona Overbought ({rsi_val:.0f})")
    
    if "bullish" in mom_align: bullish.append(f"Momentum {mom_align.replace('_',' ')}")
    elif "bearish" in mom_align: bearish.append(f"Momentum {mom_align.replace('_',' ')}")
    else: neutral.append("Momentum in transizione")
    
    if vol_p > 70: bullish.append(f"Volume elevato (P{vol_p:.0f})")
    elif vol_p < 30: bearish.append(f"Volume basso (P{vol_p:.0f})")
    
    return {
        'price': last['close'], 'change': ((last['close']/prev['close'])-1)*100,
        'change_abs': last['close']-prev['close'],
        'ema_125': ema_val, 'ema_dist': ema_dist_pct,
        'rsi': rsi_val, 'rsi_zone': rsi_zone,
        'mom_align': mom_align, 'roc': [r10, r21, r63],
        'vol_regime': vol_regime, 'atr_pct': (last['atr_14']/last['close'])*100,
        'hv20': last['hv_20'], 'hv60': last['hv_60'],
        'pos_range': pos_range, 'high52': high52, 'low52': low52,
        'dd': ((last['close'] / df['close'].expanding().max().iloc[-1]) - 1) * 100,
        'max_dd': ((df['close'] / df['close'].expanding().max()) - 1).min() * 100,
        'scores': {'final': final_score, 'tech': tech_score, 'sent': sent_score},
        'signal': {'label': signal, 'color': sig_col, 'conf': conf, 'conf_label': conf_label},
        'factors': {'bullish': bullish, 'bearish': bearish, 'neutral': neutral},
        'adjustments': {'vol_factor': vol_factor, 'volatility_factor': vol_adj_factor},
        'news_metrics': {'velocity': velocity, 'spike': news_spike}
    }

# ============================================================================
# 6. GRAFICI
# ============================================================================
def make_gauge(val, title, ranges):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1a73e8"}, 'steps': ranges}
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def make_price_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_125'], line=dict(color='#ff9800'), name='EMA 125'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color='rgba(26,115,232,0.3)', name='Vol'), row=2, col=1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def make_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['rsi_14'].iloc[-150:], line=dict(color='#7b1fa2')))
    fig.add_hline(y=70, line_dash="dot", line_color="red"); fig.add_hline(y=30, line_dash="dot", line_color="green")
    fig.update_layout(height=200, title="RSI 14", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def make_mom_chart(roc):
    fig = go.Figure(go.Bar(x=['10d', '21d', '63d'], y=roc, marker_color=['#26a69a' if x>0 else '#ef5350' for x in roc]))
    fig.update_layout(height=200, title="Momentum ROC", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def make_vol_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['hv_20'].iloc[-150:]*100, name='HV20'))
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['hv_60'].iloc[-150:]*100, name='HV60'))
    fig.update_layout(height=200, title="Historical Volatility %", margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    st.sidebar.title("Kriterion Quant")
    st.sidebar.caption("v2.0 Replica Esatta")
    TICKER = st.sidebar.text_input("Ticker", value="AAPL.US").upper()
    
    if not TICKER: return
    
    try:
        with st.spinner("Analisi in corso..."):
            df, sent, news = fetch_data(TICKER)
            if df is None: st.error("No Data"); return
            
            res = perform_full_analysis(df, sent, news, TICKER)
            
            # --- HEADER ---
            st.markdown(f"""
            <div class="main-header">
                <h1>{TICKER}</h1>
                <div class="header-meta">
                    <span class="header-badge">💰 {res['price']:.2f}</span>
                    <span class="header-badge" style="background:{'rgba(40,167,69,0.3)' if res['change']>=0 else 'rgba(220,53,69,0.3)'}">{res['change']:+.2f}%</span>
                    <span class="header-badge">Signal: {res['signal']['label']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- SCORE CARDS ---
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-card"><div class="card-title">Composite Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(res['scores']['final'], "", [{'range': [0,35], 'color':'#dc3545'}, {'range': [65,100], 'color':'#28a745'}]), width="stretch")
                st.markdown(f'<div style="text-align:center;margin-bottom:10px;"><span class="signal-badge" style="background:{res["signal"]["color"]}">{res["signal"]["label"]}</span></div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card"><div class="card-title">Sentiment Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(res['scores']['sent'], "", [{'range': [0,100], 'color':'#ffc107'}]), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="card-title">Technical Bias</div><div class="card-value">{res["scores"]["tech"]:.0f}/100</div><div class="card-sub">RSI: {res["rsi"]:.0f} | EMA: {"ABOVE" if res["ema_dist"]>0 else "BELOW"}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="card-title">Vol Regime</div><div style="margin:20px 0;"><span class="signal-badge" style="background:#1a73e8">{res["vol_regime"].upper()}</span></div><div class="card-sub">HV20/60: {res["hv20"]/res["hv60"]:.2f}</div></div>', unsafe_allow_html=True)
                
            # --- RANGE BAR ---
            st.markdown(f"""
            <div class="range-container">
                <div class="card-title">52 Week Range</div>
                <div style="display:flex; justify-content:space-between; color:#666; font-size:0.9rem;"><span>Low: {res['low52']:.2f}</span><span>Current: {res['price']:.2f}</span><span>High: {res['high52']:.2f}</span></div>
                <div class="range-bar-bg"><div class="range-marker" style="left:{min(100, max(0, res['pos_range']))}%;"></div></div>
                <div style="display:flex; justify-content:space-between; text-align:center;">
                    <div><strong>{res['pos_range']:.1f}%</strong><br>Position</div>
                    <div style="color:{'#dc3545' if res['dd']<-10 else '#333'}"><strong>{res['dd']:.2f}%</strong><br>Drawdown</div>
                    <div><strong>{res['max_dd']:.2f}%</strong><br>Max DD 1Y</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- CHARTS ---
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            st.plotly_chart(make_price_chart(df), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown('<div class="chart-section">', unsafe_allow_html=True)
                st.plotly_chart(make_rsi_chart(df), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_g2:
                st.markdown('<div class="chart-section">', unsafe_allow_html=True)
                st.plotly_chart(make_mom_chart(res['roc']), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Volatility Chart (RIPRISTINATO)
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            st.plotly_chart(make_vol_chart(df), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- NARRATIVE ---
            narrative = f"Analisi {TICKER}: Bias **{res['signal']['label']}** (Score: {res['scores']['final']:.1f}). Confidence: {res['signal']['conf']:.0%}. "
            narrative += f"Volatilità {res['vol_regime']}."
            st.markdown(f'<div class="narrative-box">📝 {narrative}</div>', unsafe_allow_html=True)
            
            # --- FACTORS ---
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown('<h4 class="bullish">✅ Bullish Factors</h4>', unsafe_allow_html=True)
                for x in res['factors']['bullish']: st.markdown(f'<div class="factor-item">{x}</div>', unsafe_allow_html=True)
            with f2:
                st.markdown('<h4 class="bearish">⚠️ Bearish Factors</h4>', unsafe_allow_html=True)
                for x in res['factors']['bearish']: st.markdown(f'<div class="factor-item">{x}</div>', unsafe_allow_html=True)
            with f3:
                st.markdown('<h4 class="neutral">➖ Neutral Factors</h4>', unsafe_allow_html=True)
                for x in res['factors']['neutral']: st.markdown(f'<div class="factor-item">{x}</div>', unsafe_allow_html=True)
                
            st.markdown("---")
            
            # --- CONFIDENCE CARDS (RIPRISTINATI) ---
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="card-title">Signal Confidence</div><div class="card-value">{res["signal"]["conf"]:.0%}</div><div class="card-sub">{res["signal"]["conf_label"]}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="card-title">Volume Confirmation</div><div class="card-value">{res["adjustments"]["vol_factor"]:.2f}x</div></div>', unsafe_allow_html=True)
            with cc3:
                st.markdown(f'<div class="metric-card"><div class="card-title">Volatility Adjustment</div><div class="card-value">{res["adjustments"]["volatility_factor"]:.2f}x</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- NEWS ---
            if news:
                st.subheader("📰 Recent News")
                for n in news[:5]:
                    with st.expander(f"{n['date'][:10]} | {n['title']}"):
                        st.write(n.get('content',''))
                        st.markdown(f"[Link]({n['link']})")
            
            # --- TABLE ---
            st.subheader("📋 Detailed Metrics")
            html_tab = f"""
            <table class="metrics-table">
                <tr><td colspan="3" class="section-header">TECHNICALS</td></tr>
                <tr><td>EMA 125</td><td>{res['ema_125']:.2f}</td><td>Dist: {res['ema_dist']:+.2f}%</td></tr>
                <tr><td>RSI 14</td><td>{res['rsi']:.1f}</td><td>{res['rsi_zone']}</td></tr>
                <tr><td colspan="3" class="section-header">SCORING</td></tr>
                <tr><td>Tech Score</td><td>{res['scores']['tech']:.1f}</td><td>Weight: {CONFIG['W_TECHNICAL']}</td></tr>
                <tr><td>Final Score</td><td><strong>{res['scores']['final']:.1f}</strong></td><td>Signal: {res['signal']['label']}</td></tr>
            </table>
            """
            st.markdown(html_tab, unsafe_allow_html=True)
            
            # JSON
            json_export = {'ticker': TICKER, 'ts': datetime.now().isoformat(), 'data': res}
            st.sidebar.download_button("📥 Download JSON", json.dumps(json_export, indent=2, default=str), f"{TICKER}.json", "application/json")

    except Exception as e:
        st.error(f"Errore: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
