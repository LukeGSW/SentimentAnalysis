import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional
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

# CSS ottimizzato per evitare conflitti di rendering
st.markdown("""
<style>
    /* Card Stile */
    div.metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    div.metric-value { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    div.metric-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Fattori */
    div.factor-box { padding: 8px 12px; border-radius: 4px; margin-bottom: 6px; font-size: 0.9rem; line-height: 1.4; }
    div.bullish { background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32; }
    div.bearish { background-color: #ffebee; color: #c62828; border-left: 4px solid #c62828; }
    div.neutral { background-color: #f5f5f5; color: #616161; border-left: 4px solid #757575; }
    
    /* Narrativa */
    div.narrative-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        color: #0d47a1;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    
    /* Override Streamlit Elements */
    .stProgress > div > div > div > div { background-color: #1a73e8; }
    h1, h2, h3 { color: #1a73e8; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP API & CONFIG
# ============================================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except:
    st.error("❌ EODHD_API_KEY mancante nei secrets. Configurala nelle impostazioni dell'app.")
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
# 3. FUNZIONI DATA FETCHING
# ============================================================================

def api_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200: return r.json()
    except Exception as e:
        st.warning(f"Errore connessione API: {e}")
    return None

@st.cache_data(ttl=3600)
def fetch_ohlcv(ticker, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    url = f"{EODHD_BASE_URL}/eod/{ticker}"
    data = api_request(url, {'api_token': EODHD_API_KEY, 'from': start.strftime('%Y-%m-%d'), 'to': end.strftime('%Y-%m-%d'), 'fmt': 'json'})
    if not data: return None
    try:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df.columns = [c.lower() for c in df.columns]
        # Pulizia dati zero volume
        return df[df['volume'] > 0]
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_sentiment(ticker):
    symbol = ticker.split('.')[0]
    end = datetime.now()
    start = end - timedelta(days=30)
    url = f"{EODHD_BASE_URL}/sentiments"
    data = api_request(url, {'api_token': EODHD_API_KEY, 's': symbol, 'from': start.strftime('%Y-%m-%d'), 'to': end.strftime('%Y-%m-%d')})
    
    if not data: return None
    
    # Logica robusta per estrarre i dati corretti dal JSON EODHD
    target_data = None
    if isinstance(data, dict):
        keys = [k for k in data.keys() if k.lower() == symbol.lower()]
        if keys: target_data = data[keys[0]]
        elif len(data) > 0: target_data = data[list(data.keys())[0]]
    
    # Se arriva una lista di dati storici, facciamo la media
    if isinstance(target_data, list):
        valid_scores = []
        for x in target_data:
            val = x.get('normalized') or x.get('score')
            if val is not None: valid_scores.append(float(val))
        
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            return {'normalized': avg_score, 'count': len(valid_scores)}
        return None
    
    return target_data

@st.cache_data(ttl=3600)
def fetch_news(ticker):
    url = f"{EODHD_BASE_URL}/news"
    return api_request(url, {'api_token': EODHD_API_KEY, 's': ticker, 'limit': 50})

# ============================================================================
# 4. CALCOLI (Logica Notebook)
# ============================================================================

def calculate_technical_indicators(df):
    # EMA
    df['ema'] = df['close'].ewm(span=CONFIG['EMA_PERIOD'], adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/CONFIG['RSI_PERIOD'], min_periods=CONFIG['RSI_PERIOD'], adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/CONFIG['RSI_PERIOD'], min_periods=CONFIG['RSI_PERIOD'], adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR & HV
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=CONFIG['ATR_PERIOD'], adjust=False).mean()
    
    log_ret = np.log(c / c.shift(1))
    df['hv20'] = log_ret.rolling(20).std() * np.sqrt(252)
    df['hv60'] = log_ret.rolling(60).std() * np.sqrt(252)
    
    # ROC
    for p in CONFIG['ROC_PERIODS']:
        df[f'roc_{p}'] = ((c - c.shift(p)) / c.shift(p)) * 100
        
    return df

def analyze_asset(df, sent_data, news_list, ticker):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Metrics
    ema_dist_pct = ((last['close'] - last['ema']) / last['ema']) * 100
    ema_pos = "ABOVE" if last['close'] > last['ema'] else "BELOW"
    
    rsi_val = last['rsi']
    rsi_zone = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
    
    # Volatility
    atr_series = df['atr'].dropna().iloc[-252:]
    # Fix per percentile se dati insufficienti
    if len(atr_series) > 10:
        atr_p = stats.percentileofscore(atr_series, last['atr'])
    else:
        atr_p = 50
    vol_regime = "EXTREME" if atr_p > 90 else "HIGH" if atr_p > 70 else "LOW" if atr_p < 30 else "NORMAL"
    
    # Momentum
    roc_vals = [last[f'roc_{p}'] for p in CONFIG['ROC_PERIODS']]
    if all(x > 0 for x in roc_vals): mom_align = "Bullish"
    elif all(x < 0 for x in roc_vals): mom_align = "Bearish"
    else: mom_align = "Transitional"
    
    # Structure
    high_52 = df['high'].iloc[-252:].max()
    low_52 = df['low'].iloc[-252:].min()
    # Safety division by zero
    rng = high_52 - low_52
    pos_range = ((last['close'] - low_52) / rng * 100) if rng > 0 else 50
    # Clamping per evitare errori st.progress
    pos_range_clamped = max(0, min(100, int(pos_range)))
    
    dd_curr = ((last['close'] / df['close'].expanding().max().iloc[-1]) - 1) * 100
    
    # Sentiment
    sent_score_norm = 50.0
    if sent_data:
        val = sent_data.get('normalized', sent_data.get('score', 0))
        # Normalizzazione -1..1 a 0..100
        try:
            val_f = float(val)
            sent_score_norm = ((max(-1, min(1, val_f)) + 1) / 2) * 100
        except: pass
        
    # News Spike
    news_spike = False
    velocity = 1.0
    if news_list:
        now = datetime.now()
        c7 = sum(1 for n in news_list if (now - pd.to_datetime(n['date'][:10])).days <= 7)
        c30 = sum(1 for n in news_list if (now - pd.to_datetime(n['date'][:10])).days <= 30)
        avg = c30 / 30 if c30 else 0
        velocity = c7 / (avg * 7) if avg else 1.0
        news_spike = velocity > CONFIG['NEWS_SPIKE_THRESHOLD']

    # Scoring
    s_ema = 50 + (min(35, ema_dist_pct * 3) if ema_dist_pct > 0 else max(-35, ema_dist_pct * 3))
    s_rsi = 50 + (rsi_val - 50)
    s_mom = 75 if mom_align == "Bullish" else 25 if mom_align == "Bearish" else 50
    
    tech_score = (max(0,min(100,s_ema)) * CONFIG['W_EMA']) + \
                 (max(0,min(100,s_rsi)) * CONFIG['W_RSI']) + \
                 (max(0,min(100,s_mom)) * CONFIG['W_MOMENTUM'])
                 
    final_sent = sent_score_norm
    if news_spike:
        final_sent += 10 if sent_score_norm > 60 else -10 if sent_score_norm < 40 else 0
        
    raw_comp = (tech_score * CONFIG['W_TECHNICAL']) + (final_sent * CONFIG['W_SENTIMENT'])
    adj_vol = 0.8 if vol_regime == "EXTREME" else 1.0
    
    final_score = max(0, min(100, raw_comp * adj_vol))
    
    if final_score >= 80: signal = "STRONG BUY"
    elif final_score >= 65: signal = "BUY"
    elif final_score >= 45: signal = "NEUTRAL"
    elif final_score >= 30: signal = "SELL"
    else: signal = "STRONG SELL"
    
    confidence = 0.5 + (0.1 if mom_align != "Transitional" else 0)
    
    # Factors
    bullish, bearish, neutral = [], [], []
    if ema_pos == "ABOVE": bullish.append(f"Prezzo sopra EMA125 (+{ema_dist_pct:.1f}%)")
    else: bearish.append(f"Prezzo sotto EMA125 ({ema_dist_pct:.1f}%)")
    if rsi_val < 30: bullish.append(f"RSI Oversold ({rsi_val:.1f})")
    elif rsi_val > 70: bearish.append(f"RSI Overbought ({rsi_val:.1f})")
    if mom_align == "Bullish": bullish.append("Momentum Bullish")
    elif mom_align == "Bearish": bearish.append("Momentum Bearish")
    else: neutral.append("Momentum in transizione")
    
    narrative = f"Analisi di {ticker}: Bias **{signal}** (Score: {final_score:.1f}). Confidence: {confidence:.0%}. "
    narrative += f"Il prezzo è al {pos_range:.1f}% del range 52w. Regime volatilità: {vol_regime}."
    
    return {
        'scores': {'final': final_score, 'tech': tech_score, 'sent': final_sent},
        'signal': signal, 'confidence': confidence,
        'metrics': {
            'close': last['close'], 'change': ((last['close']/prev['close'])-1)*100,
            'ema_dist': ema_dist_pct, 'rsi': rsi_val, 'rsi_zone': rsi_zone,
            'vol_regime': vol_regime, 'pos_range': pos_range, 'pos_range_clamped': pos_range_clamped,
            'dd': dd_curr, 'hv_ratio': (last['hv20']/last['hv60'] if last['hv60'] else 1),
            'mom_align': mom_align, 'roc': roc_vals
        },
        'factors': {'bullish': bullish, 'bearish': bearish, 'neutral': neutral},
        'narrative': narrative,
        'news_metrics': {'velocity': velocity, 'spike': news_spike}
    }

# ============================================================================
# 5. APP MAIN UI
# ============================================================================

def main():
    st.sidebar.title("Kriterion Quant")
    st.sidebar.caption("Financial Sentiment Dashboard v2.0")
    
    with st.sidebar.expander("ℹ️ Guida Ticker"):
        st.markdown("- **US:** AAPL.US\n- **EU:** ENI.MI, SAP.F\n- **Crypto:** BTC-USD.CC")

    TICKER = st.sidebar.text_input("Ticker:", value="AAPL.US").upper().strip()

    if TICKER:
        try:
            with st.spinner(f"Analisi di {TICKER} in corso..."):
                # Fetch Data
                df = fetch_ohlcv(TICKER, CONFIG['MIN_TRADING_DAYS'] + 100)
                if df is None or len(df) < 50:
                    st.error(f"Dati non disponibili o insufficienti per {TICKER}. Verifica il suffisso.")
                    return

                # Calculations
                df = calculate_technical_indicators(df)
                sent = fetch_sentiment(TICKER)
                news = fetch_news(TICKER)
                res = analyze_asset(df, sent, news, TICKER)
                
                # --- HEADER ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Asset", TICKER)
                c2.metric("Prezzo", f"{res['metrics']['close']:.2f}", f"{res['metrics']['change']:+.2f}%")
                c3.metric("Signal", res['signal'], f"Conf: {res['confidence']:.0%}")
                c4.metric("Volatilità", res['metrics']['vol_regime'], f"HV Ratio: {res['metrics']['hv_ratio']:.2f}")
                
                st.markdown("---")
                
                # --- CARDS ---
                # Usiamo HTML puro per il layout "maniacale" richiesto
                st.markdown(f"""
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-label">Composite Score</div>
                        <div class="metric-value">{res['scores']['final']:.1f}</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-label">Sentiment Score</div>
                        <div class="metric-value">{res['scores']['sent']:.1f}</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-label">Technical Bias</div>
                        <div class="metric-value">{res['scores']['tech']:.1f}</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-label">Momentum</div>
                        <div class="metric-value" style="font-size: 1.4rem;">{res['metrics']['mom_align']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 52 Week Range Position")
                # Safety check: clampiamo il valore per evitare crash st.progress
                st.progress(res['metrics']['pos_range_clamped'])
                r1, r2, r3 = st.columns([1,8,1])
                r1.caption("Low")
                r2.caption(f"Position: {res['metrics']['pos_range']:.1f}%")
                r3.caption("High")
                
                st.markdown(f"<div class='narrative-box'>📝 {res['narrative']}</div>", unsafe_allow_html=True)
                
                # --- CHARTS TABS ---
                t1, t2 = st.tabs(["📉 Price & Indicators", "📊 Analysis Details"])
                
                with t1:
                    # FIX DEPRECATION: uso width="stretch" invece di use_container_width=True
                    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                    fig_p.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#ff9800'), name='EMA 125'), row=1, col=1)
                    fig_p.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color='rgba(26, 115, 232, 0.3)', name='Vol'), row=2, col=1)
                    fig_p.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_p, width="stretch")
                    
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index[-120:], y=df['rsi'].iloc[-120:], line=dict(color='#7b1fa2')))
                        fig_rsi.add_hline(y=70, line_dash='dot', line_color='red')
                        fig_rsi.add_hline(y=30, line_dash='dot', line_color='green')
                        fig_rsi.update_layout(height=250, title="RSI 14", margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_rsi, width="stretch")
                    with cc2:
                        fig_roc = go.Figure(go.Bar(x=['10d', '21d', '63d'], y=res['metrics']['roc'], marker_color=['#2e7d32' if x>0 else '#c62828' for x in res['metrics']['roc']]))
                        fig_roc.update_layout(height=250, title="Momentum ROC", margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_roc, width="stretch")

                with t2:
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1: 
                        st.markdown("#### ✅ Bullish")
                        for f in res['factors']['bullish']: st.markdown(f"<div class='factor-box bullish'>{f}</div>", unsafe_allow_html=True)
                    with col_f2:
                        st.markdown("#### ⚠️ Bearish")
                        for f in res['factors']['bearish']: st.markdown(f"<div class='factor-box bearish'>{f}</div>", unsafe_allow_html=True)
                    with col_f3:
                        st.markdown("#### ➖ Neutral")
                        for f in res['factors']['neutral']: st.markdown(f"<div class='factor-box neutral'>{f}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Detailed Metrics Table")
                    dm1, dm2 = st.columns(2)
                    with dm1:
                        st.write(f"**EMA Distance**: {res['metrics']['ema_dist']:.2f}%")
                        st.write(f"**RSI Level**: {res['metrics']['rsi']:.1f}")
                        st.write(f"**Drawdown**: {res['metrics']['dd']:.2f}%")
                    with dm2:
                        st.write(f"**News Spike**: {'YES' if res['news_metrics']['spike'] else 'No'}")
                        st.write(f"**News Velocity**: {res['news_metrics']['velocity']:.2f}x")
                        st.write(f"**HV Ratio**: {res['metrics']['hv_ratio']:.2f}")

                # JSON Export
                json_data = {
                    'meta': {'ticker': TICKER, 'ts': datetime.now().isoformat()},
                    'data': res
                }
                st.sidebar.download_button("📥 Download JSON LLM", json.dumps(json_data, indent=2), f"{TICKER}_analysis.json", "application/json")
                
                # News
                if news:
                    st.subheader("📰 Recent News")
                    for n in news[:5]:
                        try:
                            # Gestione sicura del sentiment
                            score = 0
                            if n.get('sentiment') and isinstance(n['sentiment'], dict):
                                score = float(n['sentiment'].get('normalized', 0))
                            color = "🟢" if score > 0.5 else "🔴" if score < -0.5 else "⚪"
                        except: color = "⚪"
                        
                        with st.expander(f"{color} {n.get('date','N/A')[:10]} | {n.get('title','No Title')}"):
                            st.write(n.get('content', 'No content.'))
                            st.markdown(f"[Leggi articolo]({n.get('link','#')})")

        except Exception as e:
            st.error(f"Errore critico nell'applicazione: {str(e)}")
            with st.expander("Dettagli errore"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
