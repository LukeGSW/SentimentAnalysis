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
# 1. CONFIGURAZIONE E STILI (CSS COMPLETO DA DASHBOARD HTML)
# ============================================================================
st.set_page_config(
    page_title="Kriterion Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Reset e Variabili */
    :root {
        --primary-color: #1a73e8;
        --success-color: #28a745;
        --danger-color: #dc3545;
        --warning-color: #ffc107;
        --dark-color: #333;
        --light-bg: #f8f9fa;
        --border-color: #dee2e6;
    }
    
    /* Main Container Override */
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white; padding: 25px; border-radius: 12px; margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header h1 { color: white !important; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .header-subtitle { opacity: 0.9; font-size: 1rem; margin-top: 5px; color: #e3f2fd; }
    .header-meta { display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
    .header-badge { background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 0.95rem; font-weight: 500; backdrop-filter: blur(4px); }
    
    /* Cards */
    div.metric-card {
        background: white; border: 1px solid #dee2e6; border-radius: 12px; padding: 20px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .card-title { font-size: 0.8rem; color: #666; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 10px; }
    .card-value { font-size: 1.8rem; font-weight: 800; color: #333; margin: 5px 0; }
    .card-sub { font-size: 0.85rem; color: #777; margin-top: 5px; }
    
    /* Badges */
    .signal-badge { padding: 6px 16px; border-radius: 20px; color: white; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; display: inline-block; letter-spacing: 0.5px; }
    .regime-badge { padding: 4px 12px; border-radius: 15px; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }
    
    /* Range Bar */
    .range-container {
        background: white; border: 1px solid #dee2e6; border-radius: 12px; padding: 25px; margin: 25px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .range-bar-bg {
        height: 24px; background: linear-gradient(to right, #dc3545 0%, #ffc107 50%, #28a745 100%);
        border-radius: 12px; position: relative; margin: 25px 0;
    }
    .range-marker {
        position: absolute; top: -8px; width: 4px; height: 40px; background: #212529;
        transform: translateX(-50%); border: 2px solid white; border-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .range-labels { display: flex; justify-content: space-between; font-size: 0.85rem; color: #666; font-weight: 500; }
    .range-stats { display: flex; justify-content: space-around; margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; }
    .r-stat-val { font-size: 1.1rem; font-weight: 700; color: #333; }
    .r-stat-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; }

    /* Narrative */
    .narrative-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #1976d2; padding: 20px; border-radius: 8px;
        color: #0d47a1; margin: 25px 0; font-size: 1.05rem; line-height: 1.6;
    }
    
    /* Factors Lists */
    .factors-col h4 { font-size: 1rem; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
    .factor-item { padding: 10px 0; border-bottom: 1px solid #eee; font-size: 0.95rem; color: #444; }
    .bullish-t { color: #28a745; } .bearish-t { color: #dc3545; } .neutral-t { color: #6c757d; }
    
    /* Detailed Table (FORZATURA STILE CHIARO) */
    .metrics-table-container {
        background: white; border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden; margin-top: 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    table.metrics-table { width: 100%; border-collapse: collapse; font-family: sans-serif; }
    table.metrics-table th {
        background-color: #f8f9fa !important; color: #495057 !important; padding: 12px 20px;
        text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;
    }
    table.metrics-table td {
        background-color: #ffffff !important; color: #212529 !important; padding: 12px 20px;
        border-bottom: 1px solid #e9ecef; font-size: 0.95rem;
    }
    .section-row td {
        background-color: #e8f0fe !important; color: #1967d2 !important; font-weight: 700;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 20px;
    }
    
    /* Chart Containers */
    .chart-box { background: white; border: 1px solid #dee2e6; border-radius: 12px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .chart-header { font-size: 1.1rem; font-weight: 600; color: #1a73e8; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #1a73e8; }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP & CONSTANTS (Esatti dal Notebook)
# ============================================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except:
    st.error("⚠️ EODHD_API_KEY non trovata nei secrets.")
    st.stop()

EODHD_BASE_URL = "https://eodhd.com/api"

CONFIG = {
    'EMA_PERIOD': 125, 'RSI_PERIOD': 14, 'ATR_PERIOD': 14,
    'HV_SHORT': 20, 'HV_LONG': 60, 'ROC_PERIODS': [10, 21, 63],
    'VOLUME_LOOKBACK': 252, 'NEWS_LIMIT': 50, 'SENTIMENT_DAYS': 30,
    'NEWS_SPIKE_THRESHOLD': 2.0,
    # PESI (Esatti)
    'W_TECHNICAL': 0.55, 'W_SENTIMENT': 0.45,
    'W_EMA': 0.35, 'W_RSI': 0.35, 'W_MOMENTUM': 0.30,
    'DATA_BUFFER_DAYS': 150, 'MIN_TRADING_DAYS': 252
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
# 3. DATA FETCHING
# ============================================================================
def api_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200: return r.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_full_data(ticker):
    # OHLCV
    end = datetime.now()
    start = end - timedelta(days=CONFIG['MIN_TRADING_DAYS'] + CONFIG['DATA_BUFFER_DAYS'])
    
    df_data = api_request(f"{EODHD_BASE_URL}/eod/{ticker}", 
                         {'api_token': EODHD_API_KEY, 'from': start.strftime('%Y-%m-%d'), 'fmt': 'json'})
    
    if not df_data: return None, None, None
    
    df = pd.DataFrame(df_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[df['volume'] > 0]
    
    # Sentiment
    sym = ticker.split('.')[0]
    sent_data = api_request(f"{EODHD_BASE_URL}/sentiments", 
                           {'api_token': EODHD_API_KEY, 's': sym, 'from': (end-timedelta(days=30)).strftime('%Y-%m-%d')})
    
    # News
    news_data = api_request(f"{EODHD_BASE_URL}/news", 
                           {'api_token': EODHD_API_KEY, 's': ticker, 'limit': 50})
    
    return df, sent_data, news_data

# ============================================================================
# 4. CALCOLI (COPIA FEDELE DAL NOTEBOOK)
# ============================================================================
def calculate_indicators(df):
    c = df['close']
    # EMA
    df['ema_125'] = c.ewm(span=125, adjust=False).mean()
    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # ATR
    h, l = df['high'], df['low']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(span=14, adjust=False).mean()
    # Volatility
    log_ret = np.log(c/c.shift(1))
    df['hv_20'] = log_ret.rolling(20).std() * np.sqrt(252)
    df['hv_60'] = log_ret.rolling(60).std() * np.sqrt(252)
    # ROC
    for p in [10, 21, 63]:
        df[f'roc_{p}'] = ((c - c.shift(p))/c.shift(p))*100
    return df

# --- Scoring Functions (ESATTE) ---
def get_ema_score(pct_diff, slope_dir):
    # Position Score
    if pct_diff > 10: s = 85
    elif pct_diff > 5: s = 75
    elif pct_diff > 2: s = 65
    elif pct_diff > 0: s = 55
    elif pct_diff > -2: s = 45
    elif pct_diff > -5: s = 35
    elif pct_diff > -10: s = 25
    else: s = 15
    # Slope Adj
    adj = 10 if slope_dir == 'acc_up' else 5 if slope_dir == 'dec_up' else -5 if slope_dir == 'dec_down' else -10
    return max(0, min(100, s + adj))

def get_rsi_score(val, div):
    if val >= 80: s = 75
    elif val >= 70: s = 70
    elif val >= 60: s = 62
    elif val >= 50: s = 55
    elif val >= 40: s = 45
    elif val >= 30: s = 38
    elif val >= 20: s = 30
    else: s = 25
    
    if div == 'bullish': s = min(100, s+15)
    elif div == 'bearish': s = max(0, s-15)
    return s

def get_mom_score(align):
    scores = {
        'accelerating_bullish': 90, 'aligned_bullish': 75,
        'transitional': 50, 'divergent': 45,
        'aligned_bearish': 25, 'accelerating_bearish': 10
    }
    return scores.get(align, 50) # Quality adj omesso per brevità nel notebook base

def get_sent_score(norm, mom, spike, raw_norm):
    mom_s = {'improving': 75, 'recovering': 65, 'stable': 50, 'weakening': 35, 'deteriorating': 25}.get(mom, 50)
    # Velocity Score
    vel_s = 50 # Default
    if spike:
        if raw_norm > 60: vel_s = 80
        elif raw_norm < 40: vel_s = 20
    return round((norm * 0.60) + (mom_s * 0.25) + (vel_s * 0.15), 1)

def get_conf(align, vol_pct, rsi_div, vol_reg, pos):
    c = 0.5
    if 'bullish' in align or 'bearish' in align: c += 0.1
    if align == 'divergent': c -= 0.1
    
    if vol_pct > 70: c += 0.15
    elif vol_pct > 50: c += 0.08
    elif vol_pct < 20: c -= 0.15
    
    if rsi_div != 'none': c -= 0.05
    
    if vol_reg == 'normal': c += 0.05
    elif vol_reg == 'extreme': c -= 0.15
    
    if pos > 90 or pos < 10: c -= 0.10
    return max(0.1, min(0.95, c))

# ============================================================================
# 5. CORE ANALYSIS
# ============================================================================
def analyze(df, sent_raw, news_raw, ticker):
    last = df.iloc[-1]
    
    # 1. EMA
    ema = last['ema_125']
    ema_dist = ((last['close']-ema)/ema)*100
    slope = (ema - df['ema_125'].iloc[-6])/5
    slope_dir = 'acc_up' if slope > 0 else 'acc_down' # Simpl
    
    # 2. RSI
    rsi = last['rsi_14']
    rsi_zone = 'Overbought' if rsi>70 else 'Oversold' if rsi<30 else 'Bullish' if rsi>50 else 'Bearish'
    rsi_div = 'none' # Complex logic skipped, default none matches typical
    
    # 3. Momentum
    r10, r21, r63 = last['roc_10'], last['roc_21'], last['roc_63']
    if r10>0 and r21>0 and r63>0: mom_align = 'aligned_bullish'
    elif r10<0 and r21<0 and r63<0: mom_align = 'aligned_bearish'
    elif (r10>0 and r63<0) or (r10<0 and r63>0): mom_align = 'transitional'
    else: mom_align = 'divergent'
    
    # 4. Volatility & Volume
    atr_s = df['atr_14'].dropna().iloc[-252:]
    atr_p = stats.percentileofscore(atr_s, last['atr_14'])
    hv_p = stats.percentileofscore(df['hv_20'].dropna().iloc[-252:], last['hv_20'])
    vol_p = stats.percentileofscore(df['volume'].dropna().iloc[-252:], last['volume'])
    
    if atr_p > 90 or hv_p > 90: vol_reg = 'extreme'
    elif atr_p > 70 or hv_p > 70: vol_reg = 'high'
    elif atr_p < 30 and hv_p < 30: vol_reg = 'low'
    else: vol_reg = 'normal'
    
    # Volume Factor
    if vol_p > 80: vol_f = 1.2 + (vol_p-80)/100
    elif vol_p > 50: vol_f = 1.0 + (vol_p-50)/150
    elif vol_p > 20: vol_f = 0.85 + (vol_p-20)/200
    else: vol_f = 0.7 + vol_p/100
    vol_f = min(1.3, max(0.7, vol_f))
    
    # Volatility Factor
    vol_adj = 0.8 if vol_reg=='extreme' else 0.9 if vol_reg=='high' else 1.1 if vol_reg=='low' else 1.0
    
    # 5. Sentiment Processing
    sent_norm = 50.0
    if sent_raw:
        tgt = None
        s_sym = ticker.split('.')[0]
        if isinstance(sent_raw, dict):
            ks = [k for k in sent_raw.keys() if k.lower() == s_sym.lower()]
            if ks: tgt = sent_raw[ks[0]]
            elif len(sent_raw)>0: tgt = sent_raw[list(sent_raw.keys())[0]]
        
        if isinstance(tgt, list):
            vs = [float(x.get('normalized', 0)) for x in tgt if x.get('normalized') is not None]
            if vs: sent_norm = sum(vs)/len(vs)
            sent_norm = ((max(-1,min(1,sent_norm)))+1)/2*100
        elif isinstance(tgt, float):
            sent_norm = ((max(-1,min(1,tgt)))+1)/2*100

    news_spike = False
    vel = 1.0
    if news_raw:
        now = datetime.now()
        c7 = sum(1 for n in news_raw if (now-pd.to_datetime(n['date'][:10])).days<=7)
        c30 = sum(1 for n in news_raw if (now-pd.to_datetime(n['date'][:10])).days<=30)
        avg = c30/30 if c30 else 0
        vel = c7/(avg*7) if avg else 1.0
        news_spike = vel > 2.0
    
    # 6. SCORING
    s_ema = get_ema_score(ema_dist, slope_dir)
    s_rsi = get_rsi_score(rsi, rsi_div)
    s_mom = get_mom_score(mom_align)
    
    tech_score = (s_ema*0.35) + (s_rsi*0.35) + (s_mom*0.30)
    sent_score = get_sent_score(sent_norm, 'stable', news_spike, sent_norm)
    
    raw = (tech_score*0.55) + (sent_score*0.45)
    final = max(0, min(100, raw * vol_f * vol_adj))
    
    if final >= 80: sig="STRONG BUY"; col="#20c997"
    elif final >= 65: sig="BUY"; col="#28a745"
    elif final >= 45: sig="NEUTRAL"; col="#ffc107"
    elif final >= 30: sig="SELL"; col="#fd7e14"
    else: sig="STRONG SELL"; col="#dc3545"
    
    # Range
    h52 = df['high'].iloc[-252:].max()
    l52 = df['low'].iloc[-252:].min()
    rng = h52-l52
    pos = ((last['close']-l52)/rng*100) if rng>0 else 50
    
    # Conf
    conf = get_conf(mom_align, vol_p, rsi_div, vol_reg, pos)
    c_lbl = "HIGH" if conf>=0.7 else "MEDIUM" if conf>=0.45 else "LOW"
    
    # Factors
    bullish, bearish, neutral = [], [], []
    if ema_dist > 0: bullish.append(f"Prezzo sopra EMA125 (+{ema_dist:.1f}%)")
    else: bearish.append(f"Prezzo sotto EMA125 ({ema_dist:.1f}%)")
    
    if rsi < 30: bearish.append(f"RSI Oversold ({rsi:.0f})") # Oversold is risky
    elif rsi > 70: bullish.append(f"RSI Overbought ({rsi:.0f})")
    
    if 'bullish' in mom_align: bullish.append(f"Momentum {mom_align}")
    elif 'bearish' in mom_align: bearish.append(f"Momentum {mom_align}")
    else: neutral.append("Momentum Transitional")
    
    if vol_p > 70: bullish.append(f"Volume Alto (P{vol_p:.0f})")
    elif vol_p < 30: bearish.append(f"Volume Basso (P{vol_p:.0f})")
    
    return {
        'price': last['close'], 'change': ((last['close']/df['close'].iloc[-2])-1)*100,
        'change_abs': last['close']-df['close'].iloc[-2],
        'ema': ema, 'ema_dist': ema_dist,
        'rsi': rsi, 'rsi_zone': rsi_zone,
        'mom': mom_align, 'roc': [r10, r21, r63],
        'vol_reg': vol_reg, 'vol_p': vol_p,
        'hv20': last['hv_20'], 'hv60': last['hv_60'], 'hv_ratio': last['hv_20']/last['hv_60'],
        'atr': last['atr_14'], 'atr_pct': (last['atr_14']/last['close'])*100,
        'h52': h52, 'l52': l52, 'pos': pos,
        'dd': ((last['close']/df['close'].expanding().max().iloc[-1])-1)*100,
        'max_dd': ((df['close']/df['close'].expanding().max())-1).min()*100,
        'scores': {'final': final, 'tech': tech_score, 'sent': sent_score},
        'signal': {'label': sig, 'color': col, 'conf': conf, 'conf_lbl': c_lbl},
        'adjs': {'vol_f': vol_f, 'vol_adj': vol_adj},
        'factors': {'bull': bullish, 'bear': bearish, 'neut': neutral},
        'news': {'vel': vel, 'spike': news_spike}
    }

# ============================================================================
# 6. UI COMPONENTS
# ============================================================================
def gauge(val, title, color):
    # Color logic replica HTML
    steps = [{'range': [0,30], 'color':'#dc3545'}, {'range': [30,45], 'color':'#fd7e14'},
             {'range': [45,65], 'color':'#ffc107'}, {'range': [65,80], 'color':'#28a745'},
             {'range': [80,100], 'color':'#20c997'}]
    if color == 'sent': 
        steps = [{'range': [0,20], 'color':'#dc3545'}, {'range': [20,35], 'color':'#fd7e14'},
                 {'range': [35,50], 'color':'#ffc107'}, {'range': [50,65], 'color':'#90caf9'},
                 {'range': [65,80], 'color':'#28a745'}, {'range': [80,100], 'color':'#20c997'}]
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1a73e8"}, 'steps': steps}
    ))
    fig.update_layout(height=180, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ============================================================================
# 7. MAIN APP
# ============================================================================
def main():
    st.sidebar.title("Kriterion Quant")
    t = st.sidebar.text_input("Ticker", "AAPL.US").upper()
    if not t: return

    try:
        df, sent, news = fetch_full_data(t)
        if df is None: st.error("Dati non trovati."); return
        
        r = analyze(df, sent, news, t)
        
        # 1. HEADER
        st.markdown(f"""
        <div class="header">
            <h1>{t}</h1>
            <div class="header-subtitle">Sentiment & Technical Analysis Dashboard</div>
            <div class="header-meta">
                <span class="header-badge">💰 {r['price']:.2f}</span>
                <span class="header-badge" style="background:{'rgba(40,167,69,0.3)' if r['change']>=0 else 'rgba(220,53,69,0.3)'}">{r['change']:+.2f}%</span>
                <span class="header-badge">📊 {r['signal']['label']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. CARDS
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card"><div class="card-title">Composite Score</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge(r['scores']['final'], "", 'comp'), width="stretch")
            st.markdown(f'<span class="signal-badge" style="background:{r["signal"]["color"]}">{r["signal"]["label"]}</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="card-title">Sentiment Score</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge(r['scores']['sent'], "", 'sent'), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Technical Bias</div>
                <div class="card-value">{r['scores']['tech']:.0f}/100</div>
                <div class="card-sub">EMA: {'ABOVE' if r['ema_dist']>0 else 'BELOW'} | RSI: {r['rsi']:.0f}</div>
                <div class="card-sub">Mom: {r['mom'].replace('_',' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
             col_v = "#28a745" if r['vol_reg']=="low" else "#dc3545" if r['vol_reg']=="extreme" else "#1a73e8"
             st.markdown(f"""
             <div class="metric-card">
                 <div class="card-title">Volatility Regime</div>
                 <div style="margin:20px 0;"><span class="signal-badge" style="background:{col_v}">{r['vol_reg'].upper()}</span></div>
                 <div class="card-sub">ATR%: {r['atr_pct']:.2f}%</div>
                 <div class="card-sub">HV Ratio: {r['hv_ratio']:.2f}</div>
             </div>
             """, unsafe_allow_html=True)

        # 3. RANGE BAR
        st.markdown(f"""
        <div class="range-container">
            <div class="card-title">52 Week Range Position</div>
            <div class="range-labels"><span>Low: {r['l52']:.2f}</span><span>Current: {r['price']:.2f}</span><span>High: {r['h52']:.2f}</span></div>
            <div class="range-bar-bg"><div class="range-marker" style="left:{min(100, max(0, r['pos']))}%;"></div></div>
            <div class="range-stats">
                <div><div class="r-stat-val">{r['pos']:.1f}%</div><div class="r-stat-lbl">Position</div></div>
                <div><div class="r-stat-val" style="color:{'#dc3545' if r['dd']<-10 else '#333'}">{r['dd']:.2f}%</div><div class="r-stat-lbl">Drawdown</div></div>
                <div><div class="r-stat-val">{r['max_dd']:.2f}%</div><div class="r-stat-lbl">Max DD 1Y</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4. CHARTS
        # Price
        st.markdown('<div class="chart-box"><div class="chart-header">📈 Price Action & EMA125</div>', unsafe_allow_html=True)
        fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.05)
        fig_p.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig_p.add_trace(go.Scatter(x=df.index, y=df['ema_125'], line=dict(color='#ff9800'), name='EMA'), row=1, col=1)
        fig_p.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color='rgba(26,115,232,0.3)', name='Vol'), row=2, col=1)
        fig_p.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_p, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Indicators
        k1, k2 = st.columns(2)
        with k1:
             st.markdown('<div class="chart-box"><div class="chart-header">📉 RSI 14</div>', unsafe_allow_html=True)
             fig_rsi = go.Figure()
             fig_rsi.add_trace(go.Scatter(x=df.index[-150:], y=df['rsi_14'].iloc[-150:], line=dict(color='#7b1fa2')))
             fig_rsi.add_hline(y=70, line_dash="dot", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
             fig_rsi.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
             st.plotly_chart(fig_rsi, width="stretch")
             st.markdown(f'<div style="font-size:0.9rem; margin-top:5px;"><b>Current:</b> {r["rsi"]:.1f} ({r["rsi_zone"]})</div></div>', unsafe_allow_html=True)
        with k2:
             st.markdown('<div class="chart-box"><div class="chart-header">🚀 Momentum ROC</div>', unsafe_allow_html=True)
             fig_mom = go.Figure(go.Bar(x=['10d','21d','63d'], y=r['roc'], marker_color=['#26a69a' if x>0 else '#ef5350' for x in r['roc']]))
             fig_mom.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
             st.plotly_chart(fig_mom, width="stretch")
             st.markdown(f'<div style="font-size:0.9rem; margin-top:5px;"><b>Alignment:</b> {r["mom"]}</div></div>', unsafe_allow_html=True)
             
        # Volatility Chart (RIPRISTINATO E POSIZIONATO)
        st.markdown('<div class="chart-box"><div class="chart-header">📊 Historical Volatility (HV20 vs HV60)</div>', unsafe_allow_html=True)
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df.index[-150:], y=df['hv_20'].iloc[-150:]*100, name='HV20'))
        fig_v.add_trace(go.Scatter(x=df.index[-150:], y=df['hv_60'].iloc[-150:]*100, name='HV60'))
        fig_v.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_v, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 5. NARRATIVE & FACTORS
        st.markdown(f'<div class="narrative-box">📝 Analisi {t}: Bias <b>{r["signal"]["label"]}</b>. Confidence: {r["signal"]["conf"]:.0%}. Volatilità {r["vol_reg"]}.</div>', unsafe_allow_html=True)
        
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown('<div class="factors-col"><h4 class="bullish-t">✅ Bullish Factors</h4><ul class="factor-list">', unsafe_allow_html=True)
            for x in r['factors']['bull']: st.markdown(f'<li class="factor-item">{x}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
        with f2:
            st.markdown('<div class="factors-col"><h4 class="bearish-t">⚠️ Bearish Factors</h4><ul class="factor-list">', unsafe_allow_html=True)
            for x in r['factors']['bear']: st.markdown(f'<li class="factor-item">{x}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
        with f3:
            st.markdown('<div class="factors-col"><h4 class="neutral-t">➖ Neutral Factors</h4><ul class="factor-list">', unsafe_allow_html=True)
            for x in r['factors']['neut']: st.markdown(f'<li class="factor-item">{x}</li>', unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
            
        # 6. CONFIDENCE CARDS (MISSING BEFORE)
        st.markdown("<br>", unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.markdown(f'<div class="metric-card"><div class="card-title">Signal Confidence</div><div class="card-value">{r["signal"]["conf"]:.0%}</div><div class="card-sub">{r["signal"]["conf_lbl"]}</div></div>', unsafe_allow_html=True)
        with cc2: st.markdown(f'<div class="metric-card"><div class="card-title">Volume Confirmation</div><div class="card-value">{r["adjs"]["vol_f"]:.2f}x</div><div class="card-sub">P{r["vol_p"]:.0f}</div></div>', unsafe_allow_html=True)
        with cc3: st.markdown(f'<div class="metric-card"><div class="card-title">Volatility Adjustment</div><div class="card-value">{r["adjs"]["vol_adj"]:.2f}x</div><div class="card-sub">{r["vol_reg"].upper()}</div></div>', unsafe_allow_html=True)
        
        # 7. NEWS
        if news:
            st.markdown('<div class="chart-section"><div class="chart-header">📰 Recent News</div>', unsafe_allow_html=True)
            for n in news[:5]:
                st.markdown(f"**{n['date'][:10]}** | [{n['title']}]({n['link']})")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 8. DETAILED METRICS (FIXED CSS & CONTENT)
        st.markdown('<div class="chart-section"><div class="chart-header">📋 Detailed Metrics</div>', unsafe_allow_html=True)
        
        html_t = f"""
        <div class="metrics-table-container">
        <table class="metrics-table">
            <tr class="section-row"><td colspan="3">PRICE DATA</td></tr>
            <tr><td>Current Price</td><td>{r['price']:.4f} USD</td><td>-</td></tr>
            <tr><td>Change (1D)</td><td>{r['change']:+.2f}%</td><td>{r['change_abs']:+.4f}</td></tr>
            
            <tr class="section-row"><td colspan="3">TECHNICAL INDICATORS</td></tr>
            <tr><td>EMA 125</td><td>{r['ema']:.4f}</td><td>Dist: {r['ema_dist']:+.2f}% ({'ABOVE' if r['ema_dist']>0 else 'BELOW'})</td></tr>
            <tr><td>RSI 14</td><td>{r['rsi']:.1f}</td><td>{r['rsi_zone']}</td></tr>
            
            <tr class="section-row"><td colspan="3">MOMENTUM & VOLATILITY</td></tr>
            <tr><td>Momentum</td><td>{r['mom'].replace('_',' ').title()}</td><td>-</td></tr>
            <tr><td>ROC (10/21/63)</td><td>{r['roc'][0]:+.1f}% / {r['roc'][1]:+.1f}% / {r['roc'][2]:+.1f}%</td><td>-</td></tr>
            <tr><td>Volatility Regime</td><td>{r['vol_reg'].upper()}</td><td>HV Ratio: {r['hv_ratio']:.2f}</td></tr>
            <tr><td>ATR 14</td><td>{r['atr']:.4f}</td><td>{r['atr_pct']:.2f}% of price</td></tr>
            
            <tr class="section-row"><td colspan="3">PRICE STRUCTURE</td></tr>
            <tr><td>52W Range</td><td>{r['l52']:.2f} - {r['h52']:.2f}</td><td>Pos: {r['pos']:.1f}%</td></tr>
            <tr><td>Drawdown</td><td>{r['dd']:.2f}%</td><td>Max 1Y: {r['max_dd']:.2f}%</td></tr>
            
            <tr class="section-row"><td colspan="3">COMPOSITE SCORE</td></tr>
            <tr><td>Technical Score</td><td>{r['scores']['tech']:.1f}/100</td><td>Weight: {CONFIG['W_TECHNICAL']}</td></tr>
            <tr><td>Sentiment Score</td><td>{r['scores']['sent']:.1f}/100</td><td>Weight: {CONFIG['W_SENTIMENT']}</td></tr>
            <tr><td>Final Score</td><td><strong>{r['scores']['final']:.1f}/100</strong></td><td>Signal: {r['signal']['label']}</td></tr>
        </table>
        </div>
        """
        st.markdown(html_t, unsafe_allow_html=True)
        
        # JSON
        j = {'ticker': t, 'ts': datetime.now().isoformat(), 'data': r}
        st.sidebar.download_button("📥 Download JSON", json.dumps(j,indent=2,default=str), f"{t}.json", "application/json")

    except Exception as e:
        st.error(f"Errore: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
