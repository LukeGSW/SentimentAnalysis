import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time

# ============================================================================
# 1. CONFIGURAZIONE PAGINA E STILE
# ============================================================================
st.set_page_config(
    page_title="Kriterion Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #1a73e8;
    }
    .badge-guide {
        background-color: #e8f0fe;
        color: #174ea6;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP API E PARAMETRI
# ============================================================================

# Recupero API Key
if "EODHD_API_KEY" not in st.secrets:
    st.error("❌ ERRORE CRITICO: EODHD_API_KEY non trovata nei secrets dell'app.")
    st.stop()

EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
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
# 3. FUNZIONI DI CALCOLO E FETCHING
# ============================================================================

def validate_ticker(ticker: str) -> Tuple[bool, str, Dict]:
    if not ticker: return False, "Ticker non inserito", {}
    if '.' not in ticker: return False, "Formato errato. Usa: SYMBOL.EXCHANGE (es. AAPL.US)", {}
    parts = ticker.rsplit('.', 1)
    if len(parts) != 2: return False, "Formato non valido", {}
    symbol, exchange = parts
    if exchange not in EXCHANGE_MAP:
        return False, f"Exchange '{exchange}' non supportato. Usa: {', '.join(EXCHANGE_MAP.keys())}", {}
    asset_info = {'symbol': symbol, 'exchange': exchange, 'full_ticker': ticker, **EXCHANGE_MAP[exchange]}
    return True, "", asset_info

def api_request_with_retry(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200: return response.json()
            elif response.status_code == 404: return None
            elif response.status_code == 429:
                time.sleep((2 ** attempt) * 2)
                continue
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None

@st.cache_data(ttl=3600)
def fetch_ohlcv_data(ticker: str, days: int) -> Optional[pd.DataFrame]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"{EODHD_BASE_URL}/eod/{ticker}"
    params = {'api_token': EODHD_API_KEY, 'from': start_date.strftime('%Y-%m-%d'), 'to': end_date.strftime('%Y-%m-%d'), 'fmt': 'json', 'order': 'a'}
    data = api_request_with_retry(url, params)
    if not data: return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.columns = [c.lower() for c in df.columns]
    df = df[df['volume'] > 0]
    return df

@st.cache_data(ttl=3600)
def fetch_sentiment_data(ticker: str, days: int) -> Optional[Dict]:
    symbol = ticker.split('.')[0]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"{EODHD_BASE_URL}/sentiments"
    params = {'api_token': EODHD_API_KEY, 's': symbol, 'from': start_date.strftime('%Y-%m-%d'), 'to': end_date.strftime('%Y-%m-%d')}
    data = api_request_with_retry(url, params)
    if not data: return None
    if isinstance(data, dict):
        if symbol.lower() in [k.lower() for k in data.keys()]:
            return data[next(k for k in data.keys() if k.lower() == symbol.lower())]
        elif len(data) > 0:
            return data[list(data.keys())[0]]
    return None

@st.cache_data(ttl=3600)
def fetch_news_data(ticker: str, limit: int) -> Optional[List[Dict]]:
    symbol = ticker.split('.')[0]
    url = f"{EODHD_BASE_URL}/news"
    params = {'api_token': EODHD_API_KEY, 's': ticker, 'limit': limit, 'offset': 0}
    data = api_request_with_retry(url, params)
    if (not data) and '.' in ticker:
        params['s'] = symbol
        data = api_request_with_retry(url, params)
    return data if data else None

# --- Indicatori Tecnici ---
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_volume_percentile(volume_series: pd.Series, lookback: int = 252) -> Tuple[float, pd.Series]:
    available_days = min(lookback, len(volume_series))
    percentiles = pd.Series(index=volume_series.index, dtype=float)
    vals = volume_series.values
    for i in range(available_days, len(vals) + 1):
        window = vals[max(0, i-available_days):i]
        percentiles.iloc[i-1] = stats.percentileofscore(window, vals[i-1])
    return percentiles.iloc[-1] if not pd.isna(percentiles.iloc[-1]) else 50.0, percentiles

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_historical_volatility(close: pd.Series, period: int) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(window=period).std() * np.sqrt(252)

def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100

def calculate_drawdown(close: pd.Series) -> pd.DataFrame:
    running_max = close.expanding().max()
    dd = close - running_max
    dd_pct = (dd / running_max) * 100
    return pd.DataFrame({'running_max': running_max, 'drawdown': dd, 'drawdown_pct': dd_pct})

# --- Helper Sentiment ---
def extract_sentiment_value(raw_value) -> float:
    """Estrae in modo sicuro un valore float dal sentiment, gestendo dict o float puri."""
    try:
        if isinstance(raw_value, dict):
            # Cerca chiavi comuni usate da EODHD
            return float(raw_value.get('normalized', raw_value.get('polarity', raw_value.get('score', 0))))
        elif raw_value is not None:
            return float(raw_value)
    except (ValueError, TypeError):
        pass
    return 0.0

def normalize_sentiment(score) -> float:
    return round(((max(-1, min(1, score)) - (-1)) / 2) * 100, 1)

# --- Helper Classificazione ---
def classify_volatility_regime(atr_p: float, hv_p: float) -> str:
    if atr_p > 90 or hv_p > 90: return "extreme"
    elif atr_p > 70 or hv_p > 70: return "high"
    elif atr_p < 30 and hv_p < 30: return "low"
    else: return "normal"

def classify_momentum(roc10, roc21, roc63):
    if roc10 > 0 and roc21 > 0 and roc63 > 0: return "aligned_bullish"
    if roc10 < 0 and roc21 < 0 and roc63 < 0: return "aligned_bearish"
    return "transitional"

def detect_rsi_divergence(price, rsi, lookback=14):
    if len(price) < lookback*2: return "none"
    price_recent = price.iloc[-lookback:]
    rsi_recent = rsi.iloc[-lookback:]
    price_prev = price.iloc[-lookback*2:-lookback]
    rsi_prev = rsi.iloc[-lookback*2:-lookback]
    
    if price_recent.min() < price_prev.min() and rsi.loc[price_recent.idxmin()] > rsi.loc[price_prev.idxmin()]:
        return "bullish"
    if price_recent.max() > price_prev.max() and rsi.loc[price_recent.idxmax()] < rsi.loc[price_prev.idxmax()]:
        return "bearish"
    return "none"

# ============================================================================
# 4. APP PRINCIPALE
# ============================================================================

st.sidebar.title("Kriterion Quant")
st.sidebar.caption("Financial Sentiment Dashboard v2.0")

# --- LEGGENDA SIDEBAR AGGIUNTA ---
with st.sidebar.expander("ℹ️ Guida Suffissi Ticker", expanded=False):
    st.markdown("""
    **Formato:** `SIMBOLO.EXCHANGE`
    
    * 🇺🇸 **US Stock:** `.US` (es. `AAPL.US`)
    * 🇮🇹 **Milano:** `.MI` (es. `ENI.MI`, `UCG.MI`)
    * 🇩🇪 **Francoforte:** `.F` (es. `SAP.F`)
    * 🇫🇷 **Parigi:** `.PA` (es. `OR.PA`)
    * 🇬🇧 **Londra:** `.L` (es. `RR.L`)
    * ₿ **Crypto:** `.CC` (es. `BTC-USD.CC`)
    * 📈 **Indici:** `.INDX` (es. `GSPC.INDX`)
    """)

st.sidebar.markdown("---")

TICKER = st.sidebar.text_input("Inserisci Ticker:", value="AAPL.US").strip().upper()

if TICKER:
    is_valid, err, asset_info = validate_ticker(TICKER)
    if not is_valid:
        st.error(f"❌ {err}")
    else:
        with st.spinner(f"Acquisizione dati per {TICKER}..."):
            # Fetch Dati
            df_ohlcv = fetch_ohlcv_data(TICKER, CONFIG['MIN_TRADING_DAYS'] + CONFIG['DATA_BUFFER_DAYS'])
            
            if df_ohlcv is None or len(df_ohlcv) < CONFIG['EMA_PERIOD']:
                st.error("Dati insufficienti per l'analisi tecnica completa.")
            else:
                sent_data = fetch_sentiment_data(TICKER, CONFIG['SENTIMENT_DAYS'])
                news_list = fetch_news_data(TICKER, CONFIG['NEWS_LIMIT'])
                
                # --- CALCOLI CORE ---
                current_price = df_ohlcv['close'].iloc[-1]
                
                # EMA
                df_ohlcv['ema_125'] = calculate_ema(df_ohlcv['close'], CONFIG['EMA_PERIOD'])
                curr_ema = df_ohlcv['ema_125'].iloc[-1]
                ema_dist_pct = ((current_price - curr_ema) / curr_ema) * 100
                
                # RSI
                df_ohlcv['rsi'] = calculate_rsi(df_ohlcv['close'], CONFIG['RSI_PERIOD'])
                curr_rsi = df_ohlcv['rsi'].iloc[-1]
                
                # Volume
                vol_pct, _ = calculate_volume_percentile(df_ohlcv['volume'], CONFIG['VOLUME_LOOKBACK'])
                
                # Volatility
                df_ohlcv['atr'] = calculate_atr(df_ohlcv, CONFIG['ATR_PERIOD'])
                df_ohlcv['hv20'] = calculate_historical_volatility(df_ohlcv['close'], CONFIG['HV_SHORT'])
                df_ohlcv['hv60'] = calculate_historical_volatility(df_ohlcv['close'], CONFIG['HV_LONG'])
                
                curr_hv20 = df_ohlcv['hv20'].iloc[-1]
                curr_hv60 = df_ohlcv['hv60'].iloc[-1]
                hv_ratio = curr_hv20 / curr_hv60 if curr_hv60 else 1.0
                
                # Calcolo percentili per regime
                atr_w = df_ohlcv['atr'].dropna().iloc[-252:]
                hv_w = df_ohlcv['hv20'].dropna().iloc[-252:]
                atr_p = stats.percentileofscore(atr_w.values, df_ohlcv['atr'].iloc[-1])
                hv_p = stats.percentileofscore(hv_w.values, curr_hv20)
                vol_regime = classify_volatility_regime(atr_p, hv_p)
                
                # Momentum
                r10 = calculate_roc(df_ohlcv['close'], 10).iloc[-1]
                r21 = calculate_roc(df_ohlcv['close'], 21).iloc[-1]
                r63 = calculate_roc(df_ohlcv['close'], 63).iloc[-1]
                mom_align = classify_momentum(r10, r21, r63)
                
                # Divergenze
                rsi_div = detect_rsi_divergence(df_ohlcv['close'], df_ohlcv['rsi'])
                
                # Price Structure
                df_an = df_ohlcv.iloc[-252:]
                high_52 = df_an['high'].max()
                low_52 = df_an['low'].min()
                pos_range = ((current_price - low_52)/(high_52 - low_52))*100
                dd_df = calculate_drawdown(df_ohlcv['close'])
                curr_dd = dd_df['drawdown_pct'].iloc[-1]
                
                # Sentiment Processing (CORRETTO TYPE ERROR)
                raw_sent_val = 0.0
                sent_score = 50.0
                if sent_data:
                    val = sent_data.get('sentiment', sent_data.get('score', sent_data.get('normalized')))
                    raw_sent_val = extract_sentiment_value(val)
                    sent_score = normalize_sentiment(raw_sent_val)
                
                # News Velocity
                news_spike = False
                news_velocity = 1.0
                if news_list:
                    now = datetime.now()
                    c7 = sum(1 for n in news_list if (now - pd.to_datetime(n['date'][:10])).days <= 7)
                    c30 = sum(1 for n in news_list if (now - pd.to_datetime(n['date'][:10])).days <= 30)
                    avg = c30 / 30 if c30 else 0
                    news_velocity = c7 / (avg * 7) if avg else 1.0
                    if news_velocity > CONFIG['NEWS_SPIKE_THRESHOLD']: news_spike = True

                # --- SCORING SYSTEM ---
                # Technical (0-100)
                s_ema = 50 + min(35, ema_dist_pct * 3) if ema_dist_pct > 0 else 50 + max(-35, ema_dist_pct * 3)
                s_rsi = 50 + (curr_rsi - 50) + (15 if rsi_div == 'bullish' else -15 if rsi_div == 'bearish' else 0)
                s_mom = 75 if 'bullish' in mom_align else 25 if 'bearish' in mom_align else 50
                
                tech_score = (max(0, min(100, s_ema)) * CONFIG['W_EMA']) + \
                             (max(0, min(100, s_rsi)) * CONFIG['W_RSI']) + \
                             (max(0, min(100, s_mom)) * CONFIG['W_MOMENTUM'])
                
                # Adjust Sentiment with Spike
                final_sent = sent_score + (10 if news_spike and sent_score > 60 else -10 if news_spike and sent_score < 40 else 0)
                final_sent = max(0, min(100, final_sent))
                
                # Composite
                raw_comp = (tech_score * CONFIG['W_TECHNICAL']) + (final_sent * CONFIG['W_SENTIMENT'])
                
                # Adjustments
                adj_vol = 0.8 if vol_regime == 'extreme' else 1.0
                adj_vlm = 1.1 if vol_pct > 80 else 0.9 if vol_pct < 20 else 1.0
                
                final_score = max(0, min(100, raw_comp * adj_vol * adj_vlm))
                
                # Signal
                if final_score >= 80: signal = "STRONG BUY"
                elif final_score >= 65: signal = "BUY"
                elif final_score >= 45: signal = "NEUTRAL"
                elif final_score >= 30: signal = "SELL"
                else: signal = "STRONG SELL"
                
                confidence = 0.5 + (0.1 if 'aligned' in mom_align else 0) + (0.1 if vol_pct > 60 else 0) - (0.15 if vol_regime == 'extreme' else 0)
                confidence = max(0.1, min(0.95, confidence))
                
                # --- EXPORT JSON ---
                json_data = {
                    "metadata": {
                        "ticker": TICKER,
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price
                    },
                    "scores": {
                        "final": round(final_score, 2),
                        "technical": round(tech_score, 2),
                        "sentiment": round(final_sent, 2),
                        "signal": signal,
                        "confidence": round(confidence, 2)
                    },
                    "indicators": {
                        "rsi": round(curr_rsi, 2),
                        "ema_125": round(curr_ema, 2),
                        "ema_dist_pct": round(ema_dist_pct, 2),
                        "volume_percentile": round(vol_pct, 1),
                        "volatility_regime": vol_regime,
                        "hv_ratio": round(hv_ratio, 2)
                    },
                    "structure": {
                        "range_52w_pos": round(pos_range, 1),
                        "drawdown": round(curr_dd, 2)
                    },
                    "momentum": {
                        "alignment": mom_align,
                        "roc": {"10d": round(r10, 2), "21d": round(r21, 2), "63d": round(r63, 2)}
                    },
                    "news_metrics": {
                        "velocity": round(news_velocity, 2),
                        "spike": news_spike
                    },
                    "llm_analysis_prompt": {
                        "context": f"Asset: {TICKER}. Price: {current_price}. Signal: {signal} (Score: {final_score:.1f}/100).",
                        "details": f"Tech Score: {tech_score:.1f}, Sentiment: {final_sent:.1f}. Volatility: {vol_regime}. Momentum: {mom_align}.",
                        "key_question": "Analizza la coerenza tra il segnale tecnico e il sentiment delle news recenti."
                    }
                }
                
                st.sidebar.download_button(
                    label="📥 Download JSON per LLM Agent",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"{TICKER}_kriterion_analysis.json",
                    mime="application/json"
                )

                # --- UI DASHBOARD ---
                # Row 1: KPI
                c1, c2, c3, c4 = st.columns(4)
                chg = ((current_price - df_ohlcv['close'].iloc[-2])/df_ohlcv['close'].iloc[-2])*100
                c1.metric("Asset", TICKER, asset_info['name'])
                c2.metric("Prezzo", f"{current_price:.2f}", f"{chg:+.2f}%")
                c3.metric("Signal", signal, f"Conf: {confidence:.0%}")
                c4.metric("Volatilità", vol_regime.upper(), f"HV Ratio: {hv_ratio:.2f}")
                
                st.markdown("---")
                
                # Row 2: Gauges & Structure (FIX: WIDTH='STRETCH' INVECE DI USE_CONTAINER_WIDTH)
                g1, g2, g3 = st.columns([1, 1, 2])
                with g1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=final_score, title={'text': "Composite Score"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1a73e8"},
                               'steps': [{'range': [0, 30], 'color': '#ef5350'}, {'range': [65, 100], 'color': '#66bb6a'}]}
                    ))
                    fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                with g2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=final_sent, title={'text': "Sentiment Score"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ab47bc"}}
                    ))
                    fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                with g3:
                    st.subheader("Posizione Range 52 Settimane")
                    st.progress(int(pos_range))
                    c_r1, c_r2, c_r3 = st.columns(3)
                    c_r1.write(f"Low: {low_52:.2f}")
                    c_r2.write(f"**{pos_range:.1f}%**")
                    c_r3.write(f"High: {high_52:.2f}")
                    st.info(f"Drawdown Attuale: {curr_dd:.2f}% | Max DD 1Y: {dd_df['drawdown_pct'].min():.2f}%")
                
                # Row 3: Charts (FIX: WIDTH='STRETCH')
                st.subheader("Analisi Tecnica")
                
                # Price Chart
                fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig_p.add_trace(go.Candlestick(x=df_ohlcv.index, open=df_ohlcv['open'], high=df_ohlcv['high'], 
                                               low=df_ohlcv['low'], close=df_ohlcv['close'], name='Price'), row=1, col=1)
                fig_p.add_trace(go.Scatter(x=df_ohlcv.index, y=df_ohlcv['ema_125'], line=dict(color='orange'), name='EMA 125'), row=1, col=1)
                colors = ['green' if c>=o else 'red' for c,o in zip(df_ohlcv['close'], df_ohlcv['open'])]
                fig_p.add_trace(go.Bar(x=df_ohlcv.index, y=df_ohlcv['volume'], marker_color=colors, name='Vol'), row=2, col=1)
                fig_p.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_p, width="stretch")
                
                # Indicators Row (FIX: WIDTH='STRETCH')
                i1, i2 = st.columns(2)
                with i1:
                    st.markdown("##### RSI & Zone")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df_ohlcv.index[-150:], y=df_ohlcv['rsi'].iloc[-150:], line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash='dot', line_color='red')
                    fig_rsi.add_hline(y=30, line_dash='dot', line_color='green')
                    fig_rsi.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_rsi, width="stretch")
                with i2:
                    st.markdown("##### Momentum ROC")
                    fig_roc = go.Figure(go.Bar(x=['10d', '21d', '63d'], y=[r10, r21, r63], 
                                              marker_color=['green' if x>0 else 'red' for x in [r10, r21, r63]]))
                    fig_roc.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_roc, width="stretch")
                
                # News (CORRETTO TYPE ERROR E LOGICA DICT)
                if news_list:
                    st.subheader(f"Ultime News ({len(news_list)})")
                    for n in news_list[:5]:
                        # Estrazione sicura del valore sentiment per l'emoji
                        sent_obj = n.get('sentiment')
                        val_float = extract_sentiment_value(sent_obj)
                        
                        emoji = "🟢" if val_float > 0.5 else "🔴" if val_float < -0.5 else "⚪"
                        
                        with st.expander(f"{emoji} {n['date'][:10]} | {n['title']}"):
                            st.write(f"Fonte: **{n['source']}**")
                            st.write(n.get('content', 'Nessun contenuto disponibile.'))
                            st.markdown(f"[Leggi articolo originale]({n['link']})")

        st.caption("Kriterion Quant Dashboard v2.0 - Powered by EODHD API")
