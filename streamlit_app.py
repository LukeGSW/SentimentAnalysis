import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time

# ============================================================================
# CONFIGURAZIONE PAGINA E STILE
# ============================================================================
st.set_page_config(
    page_title="Kriterion Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS per avvicinarsi allo stile del notebook
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric {
        background-color: transparent !important;
    }
    h1, h2, h3 {
        color: #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SETUP PARAMETRI E API
# ============================================================================

# Recupero API Key dai secrets di Streamlit
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except FileNotFoundError:
    st.error("API Key non trovata. Configura .streamlit/secrets.toml con EODHD_API_KEY.")
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
# FUNZIONI CORE (IDENTICHE AL NOTEBOOK)
# ============================================================================

def validate_ticker(ticker: str) -> Tuple[bool, str, Dict]:
    if not ticker:
        return False, "Ticker non inserito", {}
    if '.' not in ticker:
        return False, "Formato non valido. Usa: SYMBOL.EXCHANGE (es. AAPL.US)", {}
    parts = ticker.rsplit('.', 1)
    if len(parts) != 2:
        return False, "Formato non valido", {}
    symbol, exchange = parts
    if exchange not in EXCHANGE_MAP:
        known_exchanges = ', '.join(EXCHANGE_MAP.keys())
        return False, f"Exchange '{exchange}' non riconosciuto. Supportati: {known_exchanges}", {}
    asset_info = {
        'symbol': symbol,
        'exchange': exchange,
        'full_ticker': ticker,
        **EXCHANGE_MAP[exchange]
    }
    return True, "", asset_info

def api_request_with_retry(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("API Key non valida o scaduta")
                return None
            elif response.status_code == 404:
                st.error(f"Asset non trovato: verifica il formato del ticker")
                return None
            elif response.status_code == 429:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None

@st.cache_data(ttl=3600)
def fetch_ohlcv_data(ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + CONFIG['DATA_BUFFER_DAYS'])
    url = f"{EODHD_BASE_URL}/eod/{ticker}"
    params = {
        'api_token': EODHD_API_KEY,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'fmt': 'json',
        'order': 'a'
    }
    data = api_request_with_retry(url, params)
    if data is None or len(data) == 0:
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.columns = [col.lower() for col in df.columns]
    df = df[df['volume'] > 0]
    return df

@st.cache_data(ttl=3600)
def fetch_sentiment_data(ticker: str, days: int = 30) -> Optional[Dict]:
    symbol = ticker.split('.')[0]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"{EODHD_BASE_URL}/sentiments"
    params = {
        'api_token': EODHD_API_KEY,
        's': symbol,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    data = api_request_with_retry(url, params)
    if data is None: return None
    if isinstance(data, dict) and symbol.lower() in [k.lower() for k in data.keys()]:
        key = next(k for k in data.keys() if k.lower() == symbol.lower())
        return data[key]
    elif isinstance(data, dict) and len(data) > 0:
        first_key = list(data.keys())[0]
        return data[first_key]
    return None

@st.cache_data(ttl=3600)
def fetch_news_data(ticker: str, limit: int = 50) -> Optional[List[Dict]]:
    symbol = ticker.split('.')[0]
    url = f"{EODHD_BASE_URL}/news"
    params = {
        'api_token': EODHD_API_KEY,
        's': ticker,
        'limit': limit,
        'offset': 0
    }
    data = api_request_with_retry(url, params)
    if (data is None or len(data) == 0) and '.' in ticker:
        params['s'] = symbol
        data = api_request_with_retry(url, params)
    if data is None or len(data) == 0:
        return None
    news_list = []
    for item in data:
        news_item = {
            'title': item.get('title', ''),
            'source': item.get('source', 'Unknown'),
            'published_at': item.get('date', ''),
            'url': item.get('link', ''),
            'sentiment': item.get('sentiment', None)
        }
        news_list.append(news_item)
    return news_list

# --- Calcoli Tecnici ---
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_percentile(volume_series: pd.Series, lookback: int = 252) -> Tuple[float, pd.Series]:
    available_days = min(lookback, len(volume_series))
    percentiles = pd.Series(index=volume_series.index, dtype=float)
    # Ottimizzazione calcolo per streamlit (vettoriale se possibile, qui loop ridotto)
    # Manteniamo logica notebook per coerenza
    for i in range(available_days, len(volume_series) + 1):
        window = volume_series.iloc[max(0, i-available_days):i]
        current_vol = volume_series.iloc[i-1]
        percentiles.iloc[i-1] = stats.percentileofscore(window.values, current_vol)
    current_percentile = percentiles.iloc[-1] if not pd.isna(percentiles.iloc[-1]) else 50.0
    return current_percentile, percentiles

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr

def calculate_historical_volatility(close: pd.Series, period: int) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(252)
    return hv

def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100

def calculate_drawdown(close: pd.Series) -> pd.DataFrame:
    running_max = close.expanding().max()
    drawdown = close - running_max
    drawdown_pct = (drawdown / running_max) * 100
    return pd.DataFrame({'running_max': running_max, 'drawdown': drawdown, 'drawdown_pct': drawdown_pct})

# --- Interpretazione e Scoring (Helper) ---
def interpret_ema_distance(distance_pct: float) -> str:
    if distance_pct > 10: return "Estensione rialzista significativa"
    elif distance_pct > 3: return "Trend rialzista sano"
    elif distance_pct > -3: return "Prossimità alla media, fase neutra"
    elif distance_pct > -10: return "Trend ribassista"
    else: return "Estensione ribassista significativa"

def interpret_rsi(rsi_value: float) -> Tuple[str, str]:
    if rsi_value > 80: return "extreme_overbought", "Rischio correzione elevato"
    elif rsi_value > 70: return "overbought", "Momentum forte ma esteso"
    elif rsi_value > 50: return "bullish", "Momentum positivo"
    elif rsi_value > 30: return "bearish", "Momentum negativo"
    elif rsi_value > 20: return "oversold", "Possibile rimbalzo"
    else: return "extreme_oversold", "Eccesso ribassista"

def interpret_volume_percentile(percentile: float) -> str:
    if percentile > 90: return "Volume eccezionale"
    elif percentile > 80: return "Volume molto alto"
    elif percentile > 60: return "Volume sopra media"
    elif percentile > 40: return "Volume normale"
    elif percentile > 20: return "Volume sotto media"
    else: return "Volume molto basso"

def classify_volatility_regime(atr_percentile: float, hv_percentile: float) -> Tuple[str, str]:
    if atr_percentile > 90 or hv_percentile > 90: return "extreme", "Volatilità estrema - cautela massima"
    elif atr_percentile > 70 or hv_percentile > 70: return "high", "Alta volatilità - breakout più affidabili"
    elif atr_percentile < 30 and hv_percentile < 30: return "low", "Bassa volatilità - range trading favorito"
    else: return "normal", "Volatilità normale"

def classify_momentum_alignment(roc_10: float, roc_21: float, roc_63: float) -> Tuple[str, str]:
    all_positive = roc_10 > 0 and roc_21 > 0 and roc_63 > 0
    all_negative = roc_10 < 0 and roc_21 < 0 and roc_63 < 0
    if all_positive:
        if roc_10 > roc_21 > roc_63: return "accelerating_bullish", "Momentum rialzista in accelerazione"
        else: return "aligned_bullish", "Momentum rialzista allineato"
    elif all_negative:
        if roc_10 < roc_21 < roc_63: return "accelerating_bearish", "Momentum ribassista in accelerazione"
        else: return "aligned_bearish", "Momentum ribassista allineato"
    else:
        if roc_10 > 0 and roc_63 < 0: return "transitional", "Transizione (breve termine positivo)"
        elif roc_10 < 0 and roc_63 > 0: return "transitional", "Transizione (breve termine negativo)"
        else: return "divergent", "Momentum divergente"

def detect_rsi_divergence(price: pd.Series, rsi: pd.Series, lookback: int = 14, tolerance: float = 0.02, volume_pct: float = 50) -> Tuple[str, str]:
    if len(price) < lookback + 5: return "none", None
    recent_price = price.iloc[-lookback:]
    # Logica semplificata per robustezza
    price_low_recent = recent_price.min()
    if len(price) > lookback * 2:
        prev_window_price = price.iloc[-lookback*2:-lookback]
        prev_window_rsi = rsi.iloc[-lookback*2:-lookback]
        if len(prev_window_price) > 0:
            prev_price_low = prev_window_price.min()
            rsi_at_prev_low = prev_window_rsi.loc[prev_window_price.idxmin()]
            rsi_at_price_low_recent = rsi.loc[recent_price.idxmin()]
            
            if price_low_recent < prev_price_low * (1 - tolerance):
                if rsi_at_price_low_recent > rsi_at_prev_low:
                    confidence = "high" if volume_pct > 60 else "medium"
                    return "bullish", confidence
            
            prev_price_high = prev_window_price.max()
            price_high_recent = recent_price.max()
            rsi_at_price_high_recent = rsi.loc[recent_price.idxmax()]
            rsi_at_prev_high = prev_window_rsi.loc[prev_window_price.idxmax()]

            if price_high_recent > prev_price_high * (1 + tolerance):
                if rsi_at_price_high_recent < rsi_at_prev_high:
                    confidence = "high" if volume_pct > 60 else "medium"
                    return "bearish", confidence
    return "none", None

def normalize_sentiment_score(score: float, min_val: float = -1, max_val: float = 1) -> float:
    if score is None: return 50.0
    score = max(min_val, min(max_val, score))
    normalized = ((score - min_val) / (max_val - min_val)) * 100
    return round(normalized, 1)

def analyze_news_velocity(news_list: List[Dict]) -> Dict:
    if not news_list:
        return {'count_24h': 0, 'count_7d': 0, 'velocity_ratio': 1.0, 'spike_detected': False}
    
    now = datetime.now()
    count_24h = 0
    count_7d = 0
    count_30d = 0
    
    for news in news_list:
        try:
            pub_date_str = news.get('published_at', '')
            if pub_date_str:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d']:
                    try:
                        pub_date = datetime.strptime(pub_date_str[:19], fmt)
                        break
                    except: continue
                else: continue
                
                days_ago = (now - pub_date).days
                if days_ago <= 1: count_24h += 1
                if days_ago <= 7: count_7d += 1
                if days_ago <= 30: count_30d += 1
        except: continue
        
    avg_daily_30d = count_30d / 30 if count_30d > 0 else 0
    expected_7d = avg_daily_30d * 7
    velocity_ratio = count_7d / expected_7d if expected_7d > 0 else 1.0
    spike_detected = velocity_ratio > CONFIG['NEWS_SPIKE_THRESHOLD']
    
    return {
        'count_24h': count_24h,
        'count_7d': count_7d,
        'count_30d': count_30d,
        'velocity_ratio': round(velocity_ratio, 2),
        'spike_detected': spike_detected
    }

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

st.sidebar.title("Kriterion Quant")
st.sidebar.markdown("### Financial Sentiment Analysis v2.0")
st.sidebar.info("Formati supportati:\n- AAPL.US (Equity US)\n- ENI.MI (Equity IT)\n- BTC-USD.CC (Crypto)\n- SPY.US (ETF)")

TICKER = st.sidebar.text_input("Inserisci Ticker:", value="AAPL.US").strip().upper()

if TICKER:
    is_valid, error_msg, ASSET_INFO = validate_ticker(TICKER)
    
    if not is_valid:
        st.error(error_msg)
    else:
        with st.spinner(f"Analisi in corso per {TICKER}..."):
            # 1. ACQUISIZIONE DATI
            ohlcv_data = fetch_ohlcv_data(TICKER, days=CONFIG['MIN_TRADING_DAYS'] + CONFIG['DATA_BUFFER_DAYS'])
            
            if ohlcv_data is None or len(ohlcv_data) < 50:
                st.error("Dati insufficienti per l'analisi.")
            else:
                sentiment_data = fetch_sentiment_data(TICKER, days=CONFIG['SENTIMENT_DAYS'])
                news_data = fetch_news_data(TICKER, limit=CONFIG['NEWS_LIMIT'])
                
                HAS_SENTIMENT = sentiment_data is not None
                HAS_NEWS = news_data is not None and len(news_data) > 0
                
                current_price = ohlcv_data['close'].iloc[-1]
                
                # 2. CALCOLO INDICATORI CORE
                # EMA
                ohlcv_data['ema_125'] = calculate_ema(ohlcv_data['close'], CONFIG['EMA_PERIOD'])
                current_ema = ohlcv_data['ema_125'].iloc[-1]
                ema_distance_pct = ((current_price - current_ema) / current_ema) * 100
                ema_slope = (current_ema - ohlcv_data['ema_125'].iloc[-6]) / 5
                
                # RSI
                ohlcv_data['rsi_14'] = calculate_rsi(ohlcv_data['close'], CONFIG['RSI_PERIOD'])
                current_rsi = ohlcv_data['rsi_14'].iloc[-1]
                rsi_zone, rsi_interp = interpret_rsi(current_rsi)
                
                # Volume
                volume_percentile, _ = calculate_volume_percentile(ohlcv_data['volume'], CONFIG['VOLUME_LOOKBACK'])
                
                # Volatility
                ohlcv_data['atr_14'] = calculate_atr(ohlcv_data, CONFIG['ATR_PERIOD'])
                ohlcv_data['hv_20'] = calculate_historical_volatility(ohlcv_data['close'], CONFIG['HV_SHORT'])
                ohlcv_data['hv_60'] = calculate_historical_volatility(ohlcv_data['close'], CONFIG['HV_LONG'])
                
                current_atr = ohlcv_data['atr_14'].iloc[-1]
                current_hv20 = ohlcv_data['hv_20'].iloc[-1]
                current_hv60 = ohlcv_data['hv_60'].iloc[-1]
                
                # Percentili Volatilità (su ultimi 252 gg)
                atr_window = ohlcv_data['atr_14'].dropna().iloc[-252:]
                atr_percentile = stats.percentileofscore(atr_window.values, current_atr)
                hv_window = ohlcv_data['hv_20'].dropna().iloc[-252:]
                hv20_percentile = stats.percentileofscore(hv_window.values, current_hv20)
                
                vol_regime, vol_interp = classify_volatility_regime(atr_percentile, hv20_percentile)
                
                # Momentum Multi-Frame
                roc_10 = calculate_roc(ohlcv_data['close'], 10).iloc[-1]
                roc_21 = calculate_roc(ohlcv_data['close'], 21).iloc[-1]
                roc_63 = calculate_roc(ohlcv_data['close'], 63).iloc[-1]
                momentum_alignment, mom_interp = classify_momentum_alignment(roc_10, roc_21, roc_63)
                
                # RSI Divergence
                rsi_div, div_conf = detect_rsi_divergence(ohlcv_data['close'], ohlcv_data['rsi_14'], volume_pct=volume_percentile)
                
                # Price Structure (52w)
                analysis_data = ohlcv_data.iloc[-252:]
                high_52w = analysis_data['high'].max()
                low_52w = analysis_data['low'].min()
                pos_in_range = ((current_price - low_52w) / (high_52w - low_52w)) * 100
                
                drawdown_df = calculate_drawdown(ohlcv_data['close'])
                current_drawdown = drawdown_df['drawdown_pct'].iloc[-1]
                
                # 3. SENTIMENT ANALYSIS
                sentiment_score_val = 50.0
                sentiment_label = "neutral"
                news_metrics = {'velocity_ratio': 1.0, 'spike_detected': False}
                
                if HAS_SENTIMENT and isinstance(sentiment_data, dict):
                    raw_score = sentiment_data.get('sentiment', sentiment_data.get('score', sentiment_data.get('normalized')))
                    if raw_score is not None:
                        sentiment_score_val = normalize_sentiment_score(float(raw_score))
                        if sentiment_score_val >= 65: sentiment_label = "bullish"
                        elif sentiment_score_val <= 35: sentiment_label = "bearish"
                
                if HAS_NEWS:
                    news_metrics = analyze_news_velocity(news_data)
                
                # 4. COMPOSITE SCORING
                # EMA Component
                ema_score = 50
                if ema_distance_pct > 0: ema_score += min(35, ema_distance_pct * 3)
                else: ema_score += max(-35, ema_distance_pct * 3)
                ema_score = max(0, min(100, ema_score))
                
                # RSI Component
                rsi_score = 50
                if current_rsi > 50: rsi_score += (current_rsi - 50)
                else: rsi_score -= (50 - current_rsi)
                if rsi_div == 'bullish': rsi_score += 15
                elif rsi_div == 'bearish': rsi_score -= 15
                rsi_score = max(0, min(100, rsi_score))
                
                # Momentum Component
                mom_score = 50
                if 'bullish' in momentum_alignment: mom_score = 75
                elif 'bearish' in momentum_alignment: mom_score = 25
                elif momentum_alignment == 'accelerating_bullish': mom_score = 90
                
                technical_score = (ema_score * CONFIG['W_EMA']) + (rsi_score * CONFIG['W_RSI']) + (mom_score * CONFIG['W_MOMENTUM'])
                
                # Sentiment Score calculation
                final_sentiment_score = sentiment_score_val # Semplificato rispetto al notebook per brevità ma logica coerente
                if news_metrics['spike_detected']:
                    if sentiment_score_val > 60: final_sentiment_score += 10
                    elif sentiment_score_val < 40: final_sentiment_score -= 10
                final_sentiment_score = max(0, min(100, final_sentiment_score))
                
                # Raw Composite
                raw_composite = (technical_score * CONFIG['W_TECHNICAL']) + (final_sentiment_score * CONFIG['W_SENTIMENT'])
                
                # Adjustments
                vol_adj = 1.0
                if vol_regime == 'extreme': vol_adj = 0.8
                
                volume_adj = 1.0
                if volume_percentile > 80: volume_adj = 1.1
                elif volume_percentile < 20: volume_adj = 0.9
                
                final_score = max(0, min(100, raw_composite * vol_adj * volume_adj))
                
                # Signal
                if final_score >= 80: signal = "STRONG BUY"
                elif final_score >= 65: signal = "BUY"
                elif final_score >= 45: signal = "NEUTRAL"
                elif final_score >= 30: signal = "SELL"
                else: signal = "STRONG SELL"
                
                # Confidence
                confidence = 0.5
                if 'aligned' in momentum_alignment: confidence += 0.1
                if volume_percentile > 60: confidence += 0.1
                if vol_regime == 'extreme': confidence -= 0.15
                confidence = max(0.1, min(0.95, confidence))
                
                # ============================================================================
                # COSTRUZIONE JSON PER EXPORT
                # ============================================================================
                json_export = {
                    'metadata': {
                        'symbol': TICKER,
                        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'price': current_price
                    },
                    'scores': {
                        'final_score': round(final_score, 1),
                        'technical_score': round(technical_score, 1),
                        'sentiment_score': round(final_sentiment_score, 1),
                        'signal': signal,
                        'confidence': round(confidence, 2)
                    },
                    'indicators': {
                        'rsi': round(current_rsi, 2),
                        'ema_125': round(current_ema, 4),
                        'ema_distance_pct': round(ema_distance_pct, 2),
                        'volume_percentile': round(volume_percentile, 1),
                        'volatility_regime': vol_regime
                    },
                    'price_structure': {
                        'position_in_range_52w': round(pos_in_range, 1),
                        'drawdown_pct': round(current_drawdown, 2)
                    },
                    'momentum': {
                        'alignment': momentum_alignment,
                        'roc_10': round(roc_10, 2),
                        'roc_63': round(roc_63, 2)
                    },
                    'news_analysis': news_metrics,
                    'llm_prompt_context': (
                        f"Analizza {TICKER}. Signal: {signal} ({final_score}/100). "
                        f"Technical Score: {technical_score:.1f}, Sentiment: {final_sentiment_score:.1f}. "
                        f"Regime Volatilità: {vol_regime}. Momentum: {momentum_alignment}. "
                        f"Il prezzo è al {pos_in_range:.1f}% del range 52w."
                    )
                }
                
                # Sidebar Download
                st.sidebar.download_button(
                    label="📥 Download JSON Analysis",
                    data=json.dumps(json_export, indent=2),
                    file_name=f"{TICKER}_analysis.json",
                    mime="application/json"
                )

                # ============================================================================
                # DASHBOARD VISUALIZATION
                # ============================================================================
                
                # HEADER
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                prev_close = ohlcv_data['close'].iloc[-2]
                change_pct = ((current_price - prev_close)/prev_close)*100
                
                col_h1.metric("Asset", TICKER, ASSET_INFO['name'])
                col_h2.metric("Prezzo", f"{current_price:.4f} {ASSET_INFO['currency']}", f"{change_pct:+.2f}%")
                col_h3.metric("Signal", signal, f"Conf: {confidence:.0%}")
                col_h4.metric("Regime Volatilità", vol_regime.upper(), f"HV Ratio: {current_hv20/current_hv60:.2f}")
                
                st.divider()
                
                # GAUGES ROW
                c1, c2, c3 = st.columns([1, 1, 2])
                
                with c1:
                    fig_comp = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = final_score,
                        title = {'text': "Composite Score"},
                        gauge = {'axis': {'range': [0, 100]},
                                 'bar': {'color': "#1a73e8"},
                                 'steps': [
                                     {'range': [0, 30], 'color': "#ef5350"},
                                     {'range': [30, 45], 'color': "#ff9800"},
                                     {'range': [45, 65], 'color': "#ffeb3b"},
                                     {'range': [65, 80], 'color': "#9ccc65"},
                                     {'range': [80, 100], 'color': "#4caf50"}]}))
                    fig_comp.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                with c2:
                    fig_sent = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = final_sentiment_score,
                        title = {'text': "Sentiment Score"},
                        gauge = {'axis': {'range': [0, 100]},
                                 'bar': {'color': "#7b1fa2"},
                                 'steps': [{'range': [0, 100], 'color': "#f3e5f5"}]}))
                    fig_sent.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
                    st.plotly_chart(fig_sent, use_container_width=True)
                
                with c3:
                    st.markdown("### 52 Week Range Position")
                    st.progress(int(pos_in_range))
                    sub_c1, sub_c2, sub_c3 = st.columns(3)
                    sub_c1.write(f"Low: {low_52w:.2f}")
                    sub_c2.write(f"Current: {pos_in_range:.1f}%")
                    sub_c3.write(f"High: {high_52w:.2f}")
                    st.caption(f"Drawdown Corrente: {current_drawdown:.2f}%")
                    
                    st.markdown("#### Key Factors")
                    if 'bullish' in momentum_alignment: st.markdown("✅ Momentum Rialzista")
                    if volume_percentile > 70: st.markdown("✅ Volume Elevato")
                    if ema_distance_pct > 0: st.markdown(f"✅ Sopra EMA 125 (+{ema_distance_pct:.1f}%)")
                    if news_metrics['spike_detected']: st.markdown("⚠️ News Spike Detected")

                # CHARTS ROW
                st.subheader("Price Action & Technicals")
                
                # Price Chart with EMA
                fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig_price.add_trace(go.Candlestick(x=ohlcv_data.index, open=ohlcv_data['open'], high=ohlcv_data['high'],
                                low=ohlcv_data['low'], close=ohlcv_data['close'], name='Price'), row=1, col=1)
                fig_price.add_trace(go.Scatter(x=ohlcv_data.index, y=ohlcv_data['ema_125'], line=dict(color='orange', width=2), name='EMA 125'), row=1, col=1)
                
                # Volume
                colors = ['green' if c >= o else 'red' for c, o in zip(ohlcv_data['close'], ohlcv_data['open'])]
                fig_price.add_trace(go.Bar(x=ohlcv_data.index, y=ohlcv_data['volume'], marker_color=colors, name='Volume'), row=2, col=1)
                fig_price.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_price, use_container_width=True)
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("RSI 14")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=ohlcv_data.index[-252:], y=ohlcv_data['rsi_14'].iloc[-252:], line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    st.info(f"RSI Zone: {rsi_zone} | Divergence: {rsi_div}")

                with col_chart2:
                    st.subheader("Multi-Timeframe Momentum (ROC)")
                    fig_roc = go.Figure()
                    periods = ['10d', '21d', '63d']
                    vals = [roc_10, roc_21, roc_63]
                    colors_roc = ['green' if v > 0 else 'red' for v in vals]
                    fig_roc.add_trace(go.Bar(x=periods, y=vals, marker_color=colors_roc))
                    fig_roc.update_layout(height=300)
                    st.plotly_chart(fig_roc, use_container_width=True)
                    st.info(f"Alignment: {momentum_alignment}")
                
                # NEWS SECTION
                if HAS_NEWS and news_data:
                    st.subheader("Ultime News")
                    for news in news_data[:5]:
                        with st.expander(f"{news['published_at'][:10]} - {news['title']}"):
                            st.write(f"Fonte: {news['source']}")
                            st.write(f"[Link articolo]({news['url']})")
                            if news['sentiment']:
                                st.caption(f"Sentiment Score: {news['sentiment']}")

        st.markdown("---")
        st.caption("Kriterion Quant Dashboard - Powered by EODHD API - Financial Advice Not Included")
