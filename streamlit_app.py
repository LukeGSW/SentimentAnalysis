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
# 1. CONFIGURAZIONE E STILI (CSS ORIGINALE INTEGRATO)
# ============================================================================
st.set_page_config(
    page_title="Kriterion Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS per replicare esattamente lo stile della dashboard HTML
st.markdown("""
<style>
    /* Generali */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .header-meta { display: flex; gap: 15px; margin-top: 10px; font-size: 0.9rem; }
    .header-badge { background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 15px; }
    
    /* Cards */
    div.metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }
    .card-title { font-size: 0.85rem; color: #666; text-transform: uppercase; margin-bottom: 10px; font-weight: 600; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #333; margin: 10px 0; }
    .card-sub { font-size: 0.85rem; color: #666; }
    .signal-badge { padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }
    
    /* Range Bar Styles */
    .range-container {
        background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 20px 0;
    }
    .range-bar-bg {
        height: 25px;
        background: linear-gradient(to right, #dc3545 0%, #ffc107 50%, #28a745 100%);
        border-radius: 12px;
        position: relative;
        margin: 15px 0;
    }
    .range-marker {
        position: absolute; top: -5px; width: 4px; height: 35px; background: #333;
        transform: translateX(-50%); border: 1px solid white;
    }
    .range-stats { display: flex; justify-content: space-between; font-size: 0.9rem; text-align: center; margin-top: 10px; }
    
    /* Narrative & Factors */
    .narrative-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #1976d2; padding: 15px; border-radius: 5px; color: #0d47a1; margin-bottom: 20px;
    }
    .factor-item { padding: 8px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
    .bullish { color: #2e7d32; } .bearish { color: #c62828; } .neutral { color: #616161; }
    
    /* Detailed Table */
    table.metrics-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    table.metrics-table th { background-color: #f8f9fa; color: #666; padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6; }
    table.metrics-table td { padding: 10px; border-bottom: 1px solid #eee; color: #333; }
    .section-header { background-color: #1a73e8; color: white; padding: 8px; font-weight: bold; text-transform: uppercase; font-size: 0.85rem; }
    
    /* Charts */
    .chart-box { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP API & CONFIG
# ============================================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except:
    st.error("❌ EODHD_API_KEY mancante nei secrets.")
    st.stop()

EODHD_BASE_URL = "https://eodhd.com/api"

CONFIG = {
    'EMA_PERIOD': 125, 'RSI_PERIOD': 14, 'ATR_PERIOD': 14,
    'HV_SHORT': 20, 'HV_LONG': 60, 'ROC_PERIODS': [10, 21, 63],
    'VOLUME_LOOKBACK': 252, 'NEWS_LIMIT': 50, 'SENTIMENT_DAYS': 30,
    'NEWS_SPIKE_THRESHOLD': 2.0, 'W_TECHNICAL': 0.55, 'W_SENTIMENT': 0.45,
    'W_EMA': 0.35, 'W_RSI': 0.35, 'W_MOMENTUM': 0.30,
    'DATA_BUFFER_DAYS': 150, 'MIN_TRADING_DAYS': 252
}

EXCHANGE_MAP = {
    'US': {'type': 'Equity', 'currency': 'USD'},
    'MI': {'type': 'Equity', 'currency': 'EUR'},
    'L': {'type': 'Equity', 'currency': 'GBP'},
    'PA': {'type': 'Equity', 'currency': 'EUR'},
    'F': {'type': 'Equity', 'currency': 'EUR'},
    'CC': {'type': 'Crypto', 'currency': 'USD'},
    'INDX': {'type': 'Index', 'currency': 'USD'}
}

# ============================================================================
# 3. FUNZIONI DATA FETCHING & CALCOLO (CORE LOGIC)
# ============================================================================

def api_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200: return r.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    # OHLCV
    end = datetime.now()
    start = end - timedelta(days=CONFIG['MIN_TRADING_DAYS'] + 200)
    url_ohlcv = f"{EODHD_BASE_URL}/eod/{ticker}"
    data_ohlcv = api_request(url_ohlcv, {'api_token': EODHD_API_KEY, 'from': start.strftime('%Y-%m-%d'), 'fmt': 'json'})
    
    if not data_ohlcv: return None, None, None
    
    df = pd.DataFrame(data_ohlcv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[df['volume'] > 0] # Filter zero volume

    # Sentiment
    symbol = ticker.split('.')[0]
    url_sent = f"{EODHD_BASE_URL}/sentiments"
    data_sent = api_request(url_sent, {'api_token': EODHD_API_KEY, 's': symbol, 'from': (end - timedelta(days=30)).strftime('%Y-%m-%d')})
    
    # News
    url_news = f"{EODHD_BASE_URL}/news"
    data_news = api_request(url_news, {'api_token': EODHD_API_KEY, 's': ticker, 'limit': 10})
    
    return df, data_sent, data_news

def process_sentiment(data_sent, symbol):
    score = 50.0
    label = "Neutral"
    if data_sent:
        target = None
        if isinstance(data_sent, dict):
            keys = [k for k in data_sent.keys() if k.lower() == symbol.lower()]
            if keys: target = data_sent[keys[0]]
            elif len(data_sent) > 0: target = data_sent[list(data_sent.keys())[0]]
        
        if isinstance(target, list):
            vals = [float(x.get('normalized', 0)) for x in target if x.get('normalized') is not None]
            if vals:
                raw_avg = sum(vals) / len(vals)
                score = ((max(-1, min(1, raw_avg)) + 1) / 2) * 100
    
    if score >= 65: label = "Bullish"
    elif score <= 35: label = "Bearish"
    return score, label

def calculate_metrics(df, sent_score, sent_label, news_data):
    # Technicals
    df['ema'] = df['close'].ewm(span=CONFIG['EMA_PERIOD'], adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta>0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = -delta.where(delta<0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain/loss
    df['rsi'] = 100 - (100/(1+rs))
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['hv20'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
    df['hv60'] = df['log_ret'].rolling(60).std() * np.sqrt(252)
    
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    
    # Current Values
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Momentum
    roc = {}
    for p in [10, 21, 63]:
        roc[p] = ((last['close'] - df['close'].shift(p).iloc[-1]) / df['close'].shift(p).iloc[-1]) * 100
    
    mom_align = "Transitional"
    if all(roc[p] > 0 for p in roc): mom_align = "Bullish Aligned"
    elif all(roc[p] < 0 for p in roc): mom_align = "Bearish Aligned"
    
    # Structure
    high52 = df['high'].iloc[-252:].max()
    low52 = df['low'].iloc[-252:].min()
    pos_range = ((last['close'] - low52) / (high52 - low52)) * 100
    pos_range = max(0, min(100, pos_range))
    
    dd_curr = ((last['close'] / df['close'].expanding().max().iloc[-1]) - 1) * 100
    max_dd = ((df['close'] / df['close'].expanding().max()) - 1).min() * 100
    
    # Volatility Regime
    atr_p = stats.percentileofscore(df['atr'].dropna().iloc[-252:], last['atr'])
    vol_regime = "Normal"
    if atr_p > 90: vol_regime = "Extreme"
    elif atr_p > 70: vol_regime = "High"
    elif atr_p < 30: vol_regime = "Low"
    
    # Volume
    vol_p = stats.percentileofscore(df['volume'].dropna().iloc[-252:], last['volume'])
    
    # Composite Score
    ema_dist = ((last['close'] - last['ema'])/last['ema'])*100
    
    s_tech = 50
    if ema_dist > 0: s_tech += 20
    else: s_tech -= 20
    if last['rsi'] < 30: s_tech += 15 # Mean rev
    elif last['rsi'] > 70: s_tech -= 15
    if "Bullish" in mom_align: s_tech += 15
    elif "Bearish" in mom_align: s_tech -= 15
    s_tech = max(0, min(100, s_tech))
    
    raw_comp = (s_tech * 0.55) + (sent_score * 0.45)
    final_score = raw_comp # Simplified
    
    signal = "NEUTRAL"
    color = "#ffc107" # yellow
    if final_score >= 65: signal = "BUY"; color = "#28a745"
    elif final_score >= 80: signal = "STRONG BUY"; color = "#20c997"
    elif final_score <= 35: signal = "SELL"; color = "#fd7e14"
    elif final_score <= 20: signal = "STRONG SELL"; color = "#dc3545"
    
    return {
        'price': last['close'], 'change': ((last['close']-prev['close'])/prev['close'])*100,
        'change_abs': last['close']-prev['close'],
        'ema': last['ema'], 'ema_dist': ema_dist,
        'rsi': last['rsi'], 'vol_p': vol_p,
        'hv20': last['hv20'], 'hv60': last['hv60'],
        'atr': last['atr'], 'atr_pct': (last['atr']/last['close'])*100,
        'high52': high52, 'low52': low52, 'pos_range': pos_range,
        'dd': dd_curr, 'max_dd': max_dd,
        'roc': roc, 'mom_align': mom_align,
        'vol_regime': vol_regime,
        'scores': {'tech': s_tech, 'sent': sent_score, 'final': final_score},
        'signal': {'label': signal, 'color': color, 'conf': 0.65}, # Fixed conf for demo
        'factors': {
            'bullish': [f"Prezzo > EMA ({ema_dist:.1f}%)" if ema_dist>0 else None, f"Volume Alto (P{vol_p:.0f})" if vol_p>70 else None],
            'bearish': [f"RSI Overbought ({last['rsi']:.0f})" if last['rsi']>70 else None, "Trend Bearish" if ema_dist<0 else None],
            'neutral': ["Volatilità Bassa" if vol_regime=="Low" else None]
        }
    }

# ============================================================================
# 4. GRAFICI PLOTLY (IDENTICI ALL'HTML)
# ============================================================================

def make_gauge(value, title, color_steps):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 14}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1a73e8"},
               'steps': color_steps}
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def make_price_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#ff9800', width=2), name='EMA 125'), row=1, col=1)
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Vol'), row=2, col=1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def make_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['rsi'].iloc[-150:], line=dict(color='#7b1fa2')))
    fig.add_hline(y=70, line_dash="dot", line_color="red"); fig.add_hline(y=30, line_dash="dot", line_color="green")
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10), title="RSI 14")
    return fig

def make_mom_chart(roc):
    fig = go.Figure(go.Bar(x=['10d', '21d', '63d'], y=[roc[10], roc[21], roc[63]], 
                           marker_color=['#26a69a' if x>0 else '#ef5350' for x in [roc[10], roc[21], roc[63]]]))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10), title="Momentum ROC")
    return fig

def make_vol_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['hv20'].iloc[-150:]*100, name='HV 20', line=dict(color='#1a73e8')))
    fig.add_trace(go.Scatter(x=df.index[-150:], y=df['hv60'].iloc[-150:]*100, name='HV 60', line=dict(color='#ff9800')))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10), title="Historical Volatility %")
    return fig

# ============================================================================
# 5. APP PRINCIPALE (LAYOUT)
# ============================================================================

def main():
    # --- SIDEBAR ---
    st.sidebar.title("Kriterion Quant")
    st.sidebar.caption("v2.0 Dashboard")
    TICKER = st.sidebar.text_input("Ticker:", value="AAPL.US").upper()
    
    if not TICKER: return

    try:
        with st.spinner("Generazione Dashboard..."):
            df, data_sent, data_news = fetch_data(TICKER)
            
            if df is None or len(df) < 100:
                st.error("Dati insufficienti o ticker non valido.")
                return
            
            # Calcoli
            sym = TICKER.split('.')[0]
            sent_score, sent_label = process_sentiment(data_sent, sym)
            res = calculate_metrics(df, sent_score, sent_label, data_news)
            
            # 1. HEADER
            suff = TICKER.split('.')[1] if '.' in TICKER else 'US'
            info = EXCHANGE_MAP.get(suff, {'type': 'Equity', 'currency': 'USD'})
            
            st.markdown(f"""
            <div class="main-header">
                <h1>{TICKER} - {info['type']}</h1>
                <div class="header-meta">
                    <span class="header-badge">📅 {df.index[-1].strftime('%Y-%m-%d')}</span>
                    <span class="header-badge">💰 {res['price']:.4f} {info['currency']}</span>
                    <span class="header-badge" style="background: {'rgba(38,166,154,0.3)' if res['change']>=0 else 'rgba(239,83,80,0.3)'}">
                        {res['change']:+.2f}% (1D)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. SCORE CARDS
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-card"><div class="card-title">Composite Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(res['scores']['final'], "", 
                    [{'range': [0, 35], 'color': '#dc3545'}, {'range': [65, 100], 'color': '#28a745'}]), width="stretch")
                st.markdown(f'<div style="text-align:center; margin-bottom:10px;"><span class="signal-badge" style="background:{res["signal"]["color"]}">{res["signal"]["label"]}</span></div></div>', unsafe_allow_html=True)
            
            with c2:
                st.markdown('<div class="metric-card"><div class="card-title">Sentiment Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(res['scores']['sent'], "", 
                    [{'range': [0, 35], 'color': '#dc3545'}, {'range': [35, 65], 'color': '#ffc107'}, {'range': [65, 100], 'color': '#28a745'}]), width="stretch")
                st.markdown(f'<div class="card-sub" style="text-align:center; margin-bottom:10px;">{sent_label}</div></div>', unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Technical Bias</div>
                    <div class="card-value">{res['scores']['tech']:.0f}/100</div>
                    <div class="card-sub">EMA: {'ABOVE' if res['ema_dist']>0 else 'BELOW'} | RSI: {res['rsi']:.0f}</div>
                    <div class="card-sub" style="margin-top:5px;">Mom: {res['mom_align']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                color_reg = "#28a745" if res['vol_regime']=="Low" else "#dc3545" if res['vol_regime']=="Extreme" else "#1a73e8"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Volatility Regime</div>
                    <div style="margin:20px 0;"><span class="signal-badge" style="background:{color_reg}">{res['vol_regime'].upper()}</span></div>
                    <div class="card-sub">ATR%: {res['atr_pct']:.2f}%</div>
                    <div class="card-sub">HV20/60: {res['hv20']/res['hv60']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. RANGE BAR 52W
            st.markdown(f"""
            <div class="range-container">
                <div class="card-title">52 Week Range Position</div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#666;">
                    <span>Low: {res['low52']:.2f}</span>
                    <span>Current: {res['price']:.2f}</span>
                    <span>High: {res['high52']:.2f}</span>
                </div>
                <div class="range-bar-bg">
                    <div class="range-marker" style="left: {res['pos_range']}%;"></div>
                </div>
                <div class="range-stats">
                    <div><strong>{res['pos_range']:.1f}%</strong><br>Position</div>
                    <div style="color:{'#dc3545' if res['dd']<-10 else '#333'}"><strong>{res['dd']:.2f}%</strong><br>Drawdown</div>
                    <div><strong>{res['max_dd']:.2f}%</strong><br>Max DD 1Y</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. CHARTS SECTION
            # Row 1: Price
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(make_price_chart(df), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 2: RSI & Momentum
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                st.plotly_chart(make_rsi_chart(df), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_ch2:
                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                st.plotly_chart(make_mom_chart(res['roc']), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Row 3: Volatility (Mancante prima)
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(make_vol_chart(df), width="stretch")
            st.markdown(f'<div style="text-align:center; font-size:0.9rem; color:#666;">HV20: {res["hv20"]*100:.1f}% | HV60: {res["hv60"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 5. NARRATIVE & FACTORS
            narrative = f"L'analisi di {TICKER} indica un bias **{res['signal']['label']}** (Score: {res['scores']['final']:.1f}). "
            narrative += f"Il prezzo si trova al {res['pos_range']:.1f}% del range annuale. "
            if res['ema_dist'] > 0: narrative += "Il trend di fondo è positivo (sopra EMA 125). "
            narrative += f"Volatilità {res['vol_regime']}."
            
            st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)
            
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown('<h4 class="bullish">✅ Bullish Factors</h4>', unsafe_allow_html=True)
                for f in [x for x in res['factors']['bullish'] if x]: st.markdown(f'<div class="factor-item">{f}</div>', unsafe_allow_html=True)
            with f2:
                st.markdown('<h4 class="bearish">⚠️ Bearish Factors</h4>', unsafe_allow_html=True)
                for f in [x for x in res['factors']['bearish'] if x]: st.markdown(f'<div class="factor-item">{f}</div>', unsafe_allow_html=True)
            with f3:
                st.markdown('<h4 class="neutral">➖ Neutral Factors</h4>', unsafe_allow_html=True)
                for f in [x for x in res['factors']['neutral'] if x]: st.markdown(f'<div class="factor-item">{f}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 6. NEWS SECTION
            if data_news:
                st.subheader("📰 Recent News")
                for n in data_news[:5]:
                    with st.expander(f"{n['date'][:10]} | {n['title']}"):
                        st.write(n.get('content', 'No content'))
                        st.markdown(f"[Link]({n['link']})")
            
            st.markdown("---")
            
            # 7. DETAILED METRICS TABLE (HTML REPLICATION)
            st.subheader("📋 Detailed Metrics")
            
            # Costruzione tabella HTML
            html_table = f"""
            <table class="metrics-table">
                <tr><td colspan="3" class="section-header">PRICE DATA</td></tr>
                <tr><td>Current Price</td><td>{res['price']:.4f} {info['currency']}</td><td>-</td></tr>
                <tr><td>Change (1D)</td><td>{res['change']:+.2f}%</td><td>{res['change_abs']:+.4f}</td></tr>
                
                <tr><td colspan="3" class="section-header">TECHNICAL INDICATORS</td></tr>
                <tr><td>EMA 125</td><td>{res['ema']:.4f}</td><td>Dist: {res['ema_dist']:+.2f}%</td></tr>
                <tr><td>RSI 14</td><td>{res['rsi']:.1f}</td><td>{'Overbought' if res['rsi']>70 else 'Oversold' if res['rsi']<30 else 'Neutral'}</td></tr>
                
                <tr><td colspan="3" class="section-header">MOMENTUM</td></tr>
                <tr><td>ROC 10d</td><td>{res['roc'][10]:+.2f}%</td><td>Short Term</td></tr>
                <tr><td>ROC 63d</td><td>{res['roc'][63]:+.2f}%</td><td>Long Term</td></tr>
                <tr><td>Alignment</td><td>{res['mom_align']}</td><td>-</td></tr>
                
                <tr><td colspan="3" class="section-header">VOLATILITY</td></tr>
                <tr><td>ATR 14</td><td>{res['atr']:.4f}</td><td>{res['atr_pct']:.2f}% of price</td></tr>
                <tr><td>HV 20</td><td>{res['hv20']*100:.1f}%</td><td>Annualized</td></tr>
                <tr><td>HV 60</td><td>{res['hv60']*100:.1f}%</td><td>Annualized</td></tr>
                <tr><td>Regime</td><td>{res['vol_regime']}</td><td>Ratio: {res['hv20']/res['hv60']:.2f}</td></tr>
                
                <tr><td colspan="3" class="section-header">PRICE STRUCTURE</td></tr>
                <tr><td>52W High</td><td>{res['high52']:.2f}</td><td>-</td></tr>
                <tr><td>52W Low</td><td>{res['low52']:.2f}</td><td>-</td></tr>
                <tr><td>Pos in Range</td><td>{res['pos_range']:.1f}%</td><td>-</td></tr>
                <tr><td>Drawdown</td><td>{res['dd']:.2f}%</td><td>Max 1Y: {res['max_dd']:.2f}%</td></tr>
                
                <tr><td colspan="3" class="section-header">COMPOSITE SCORE</td></tr>
                <tr><td>Technical</td><td>{res['scores']['tech']:.1f}/100</td><td>Weight: {CONFIG['W_TECHNICAL']}</td></tr>
                <tr><td>Sentiment</td><td>{res['scores']['sent']:.1f}/100</td><td>Weight: {CONFIG['W_SENTIMENT']}</td></tr>
                <tr><td><strong>Final Score</strong></td><td><strong>{res['scores']['final']:.1f}/100</strong></td><td>Signal: {res['signal']['label']}</td></tr>
            </table>
            """
            st.markdown(html_table, unsafe_allow_html=True)
            
            # JSON Export
            json_export = {
                'ticker': TICKER,
                'timestamp': datetime.now().isoformat(),
                'metrics': res
            }
            st.sidebar.download_button("📥 Download JSON", json.dumps(json_export, indent=2, default=str), f"{TICKER}_analysis.json", "application/json")

    except Exception as e:
        st.error(f"Errore critico: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
