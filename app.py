# -*- coding: utf-8 -*-
"""
Financial Sentiment Analysis Dashboard - Streamlit App
Kriterion Quant v2.0

Convertito da Colab Notebook per deployment su Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import warnings
import time

warnings.filterwarnings('ignore')

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats

# =============================================================================
# CONFIGURAZIONE PAGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Kriterion Quant - Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURAZIONE API E PARAMETRI
# =============================================================================

# API Key da Streamlit Secrets
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    EODHD_API_KEY = None

EODHD_BASE_URL = "https://eodhd.com/api"

# Parametri Indicatori
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

# Mapping Exchange
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


# =============================================================================
# FUNZIONI DI VALIDAZIONE
# =============================================================================

def validate_ticker(ticker: str) -> Tuple[bool, str, Dict]:
    """Valida il formato del ticker e identifica il tipo di asset."""
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


# =============================================================================
# FUNZIONI API EODHD
# =============================================================================

def api_request_with_retry(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    """Esegue richiesta API con retry e backoff esponenziale."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("❌ API Key non valida o scaduta")
                return None
            elif response.status_code == 404:
                st.error("❌ Asset non trovato: verifica il formato del ticker")
                return None
            elif response.status_code == 429:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"⚠️ Errore HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            time.sleep(2)
    
    return None


def fetch_ohlcv_data(ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
    """Recupera dati OHLCV storici da EODHD."""
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
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    df = df[df['volume'] > 0]
    
    return df


def fetch_sentiment_data(ticker: str, days: int = 30) -> Optional[Dict]:
    """Recupera dati sentiment aggregati da EODHD."""
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
    
    if data is None:
        return None
    
    if isinstance(data, dict) and symbol.lower() in [k.lower() for k in data.keys()]:
        key = next(k for k in data.keys() if k.lower() == symbol.lower())
        return data[key]
    elif isinstance(data, dict) and len(data) > 0:
        first_key = list(data.keys())[0]
        return data[first_key]
    
    return None


def fetch_news_data(ticker: str, limit: int = 50) -> Optional[List[Dict]]:
    """Recupera news recenti da EODHD."""
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


# =============================================================================
# FUNZIONI CALCOLO INDICATORI
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcola Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcola RSI con metodo Wilder smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volume_percentile(volume_series: pd.Series, lookback: int = 252) -> Tuple[float, pd.Series]:
    """Calcola il percentile del volume corrente nella distribuzione storica."""
    available_days = min(lookback, len(volume_series))
    percentiles = pd.Series(index=volume_series.index, dtype=float)
    
    for i in range(available_days, len(volume_series) + 1):
        window = volume_series.iloc[max(0, i-available_days):i]
        current_vol = volume_series.iloc[i-1]
        percentiles.iloc[i-1] = stats.percentileofscore(window.values, current_vol)
    
    current_percentile = percentiles.iloc[-1] if not pd.isna(percentiles.iloc[-1]) else 50.0
    return current_percentile, percentiles


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcola Average True Range."""
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
    """Calcola Historical Volatility annualizzata."""
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(252)
    return hv


def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    """Calcola Rate of Change."""
    return ((series - series.shift(period)) / series.shift(period)) * 100


def calculate_drawdown(close: pd.Series) -> pd.DataFrame:
    """Calcola serie drawdown dal running maximum."""
    running_max = close.expanding().max()
    drawdown = close - running_max
    drawdown_pct = (drawdown / running_max) * 100
    return pd.DataFrame({
        'running_max': running_max,
        'drawdown': drawdown,
        'drawdown_pct': drawdown_pct
    })


# =============================================================================
# FUNZIONI DI INTERPRETAZIONE
# =============================================================================

def interpret_ema_distance(distance_pct: float) -> str:
    """Interpreta distanza percentuale da EMA."""
    if distance_pct > 10:
        return "Estensione rialzista significativa"
    elif distance_pct > 3:
        return "Trend rialzista sano"
    elif distance_pct > -3:
        return "Prossimità alla media, fase neutra"
    elif distance_pct > -10:
        return "Trend ribassista"
    else:
        return "Estensione ribassista significativa"


def interpret_rsi(rsi_value: float) -> Tuple[str, str]:
    """Interpreta valore RSI."""
    if rsi_value > 80:
        return "extreme_overbought", "Rischio correzione elevato"
    elif rsi_value > 70:
        return "overbought", "Momentum forte ma esteso"
    elif rsi_value > 50:
        return "bullish", "Momentum positivo"
    elif rsi_value > 30:
        return "bearish", "Momentum negativo"
    elif rsi_value > 20:
        return "oversold", "Possibile rimbalzo"
    else:
        return "extreme_oversold", "Eccesso ribassista"


def interpret_volume_percentile(percentile: float) -> str:
    """Interpreta percentile volume."""
    if percentile > 90:
        return "Volume eccezionale, evento significativo"
    elif percentile > 80:
        return "Volume molto alto, conferma forte"
    elif percentile > 60:
        return "Volume sopra media"
    elif percentile > 40:
        return "Volume normale"
    elif percentile > 20:
        return "Volume sotto media"
    else:
        return "Volume molto basso, movimento poco affidabile"


def classify_volatility_regime(atr_percentile: float, hv_percentile: float) -> Tuple[str, str]:
    """Classifica il regime di volatilità."""
    if atr_percentile > 90 or hv_percentile > 90:
        return "extreme", "Volatilità estrema - cautela massima sui segnali"
    elif atr_percentile > 70 or hv_percentile > 70:
        return "high", "Alta volatilità - breakout più affidabili, mean reversion rischiosa"
    elif atr_percentile < 30 and hv_percentile < 30:
        return "low", "Bassa volatilità - range trading favorito, breakout meno affidabili"
    else:
        return "normal", "Volatilità normale - segnali standard affidabili"


def interpret_hv_ratio(ratio: float) -> str:
    """Interpreta il rapporto HV20/HV60."""
    if ratio > 1.3:
        return "Volatilità in forte espansione"
    elif ratio > 1.1:
        return "Volatilità in espansione"
    elif ratio > 0.9:
        return "Volatilità stabile"
    elif ratio > 0.7:
        return "Volatilità in contrazione"
    else:
        return "Volatilità in forte contrazione"


def classify_momentum_alignment(roc_10: float, roc_21: float, roc_63: float) -> Tuple[str, str]:
    """Classifica l'allineamento del momentum multi-timeframe."""
    all_positive = roc_10 > 0 and roc_21 > 0 and roc_63 > 0
    all_negative = roc_10 < 0 and roc_21 < 0 and roc_63 < 0
    
    if all_positive:
        if roc_10 > roc_21 > roc_63:
            return "accelerating_bullish", "Momentum rialzista in accelerazione su tutti i timeframe"
        else:
            return "aligned_bullish", "Momentum rialzista allineato su tutti i timeframe"
    elif all_negative:
        if roc_10 < roc_21 < roc_63:
            return "accelerating_bearish", "Momentum ribassista in accelerazione su tutti i timeframe"
        else:
            return "aligned_bearish", "Momentum ribassista allineato su tutti i timeframe"
    else:
        if roc_10 > 0 and roc_63 < 0:
            return "transitional", "Fase di transizione: breve termine positivo, lungo termine negativo"
        elif roc_10 < 0 and roc_63 > 0:
            return "transitional", "Fase di transizione: breve termine negativo, lungo termine positivo"
        else:
            return "divergent", "Momentum divergente tra timeframe"


def detect_rsi_divergence(price: pd.Series, rsi: pd.Series, volume_percentile: float, lookback: int = 14, tolerance: float = 0.02) -> Tuple[str, Optional[str]]:
    """Rileva divergenze RSI."""
    if len(price) < lookback + 5:
        return "none", None
    
    recent_price = price.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]
    
    price_low_recent = recent_price.min()
    
    if len(price) > lookback * 2:
        prev_window_price = price.iloc[-lookback*2:-lookback]
        prev_window_rsi = rsi.iloc[-lookback*2:-lookback]
        
        if len(prev_window_price) > 0 and len(prev_window_rsi) > 0:
            prev_price_low = prev_window_price.min()
            rsi_at_price_low_recent = rsi.loc[recent_price.idxmin()]
            rsi_at_prev_low = prev_window_rsi.loc[prev_window_price.idxmin()]
            
            # Bullish Divergence
            if price_low_recent < prev_price_low * (1 - tolerance):
                if rsi_at_price_low_recent > rsi_at_prev_low:
                    confidence = "high" if volume_percentile > 60 else "medium"
                    return "bullish", confidence
            
            # Bearish Divergence
            prev_price_high = prev_window_price.max()
            price_high_recent = recent_price.max()
            rsi_at_price_high_recent = rsi.loc[recent_price.idxmax()]
            rsi_at_prev_high = prev_window_rsi.loc[prev_window_price.idxmax()]
            
            if price_high_recent > prev_price_high * (1 + tolerance):
                if rsi_at_price_high_recent < rsi_at_prev_high:
                    confidence = "high" if volume_percentile > 60 else "medium"
                    return "bearish", confidence
    
    return "none", None


def assess_momentum_quality(alignment: str, divergence: str, volume_pct: float) -> str:
    """Valuta la qualità complessiva del momentum."""
    is_aligned = alignment in ['aligned_bullish', 'aligned_bearish', 'accelerating_bullish', 'accelerating_bearish']
    has_contrary_divergence = (
        (alignment.endswith('bullish') and divergence == 'bearish') or
        (alignment.endswith('bearish') and divergence == 'bullish')
    )
    
    if is_aligned and divergence == 'none' and volume_pct > 50:
        return "strong"
    elif is_aligned and not has_contrary_divergence:
        return "moderate"
    elif has_contrary_divergence:
        return "conflicting"
    else:
        return "weak"


def interpret_range_position(position_pct: float) -> Tuple[str, str]:
    """Interpreta posizione nel range 52w."""
    if position_pct > 90:
        return "near_high", "Prossimo a resistenza annuale - cautela sui long"
    elif position_pct > 70:
        return "upper_range", "Forza relativa - trend rialzista intatto"
    elif position_pct > 30:
        return "mid_range", "Zona neutra - equilibrio domanda/offerta"
    elif position_pct > 10:
        return "lower_range", "Debolezza relativa - possibile supporto in avvicinamento"
    else:
        return "near_low", "Prossimo a supporto annuale - potenziale rimbalzo"


def interpret_drawdown(dd_pct: float) -> str:
    """Interpreta drawdown corrente."""
    if dd_pct == 0:
        return "All-time high del periodo"
    elif dd_pct > -5:
        return "Correzione minima"
    elif dd_pct > -10:
        return "Correzione moderata"
    elif dd_pct > -20:
        return "Correzione significativa"
    else:
        return "Bear market territory"


def interpret_single_roc(roc_val: float, period: int) -> str:
    """Interpreta singolo valore ROC."""
    timeframe = "2 settimane" if period == 10 else "1 mese" if period == 21 else "1 trimestre"
    if roc_val > 10:
        return f"Forte rialzo su {timeframe}"
    elif roc_val > 3:
        return f"Rialzo moderato su {timeframe}"
    elif roc_val > -3:
        return f"Laterale su {timeframe}"
    elif roc_val > -10:
        return f"Ribasso moderato su {timeframe}"
    else:
        return f"Forte ribasso su {timeframe}"


# =============================================================================
# FUNZIONI SENTIMENT PROCESSING
# =============================================================================

def normalize_sentiment_score(score: float, min_val: float = -1, max_val: float = 1) -> float:
    """Normalizza sentiment score a scala 0-100."""
    if score is None:
        return 50.0
    score = max(min_val, min(max_val, score))
    normalized = ((score - min_val) / (max_val - min_val)) * 100
    return round(normalized, 1)


def classify_sentiment_label(normalized_score: float) -> str:
    """Classifica sentiment in label testuale."""
    if normalized_score >= 80:
        return "very_bullish"
    elif normalized_score >= 65:
        return "bullish"
    elif normalized_score >= 55:
        return "slightly_bullish"
    elif normalized_score >= 45:
        return "neutral"
    elif normalized_score >= 35:
        return "slightly_bearish"
    elif normalized_score >= 20:
        return "bearish"
    else:
        return "very_bearish"


def classify_sentiment_momentum(delta_7d: float, delta_30d: float) -> Tuple[str, str]:
    """Classifica momentum del sentiment."""
    threshold = 5
    
    if delta_7d > threshold and delta_30d > threshold:
        return "improving", "Sentiment in miglioramento costante"
    elif delta_7d > threshold and delta_30d < -threshold:
        return "recovering", "Sentiment in recupero da livelli bassi"
    elif delta_7d < -threshold and delta_30d > threshold:
        return "weakening", "Sentiment in indebolimento dopo periodo positivo"
    elif delta_7d < -threshold and delta_30d < -threshold:
        return "deteriorating", "Sentiment in deterioramento costante"
    else:
        return "stable", "Sentiment stabile"


def analyze_news_velocity(news_list: List[Dict], days_7: int = 7, days_30: int = 30) -> Dict:
    """Analizza velocità e distribuzione delle news."""
    if not news_list:
        return {
            'count_24h': 0,
            'count_7d': 0,
            'count_30d': 0,
            'avg_daily_30d': 0,
            'velocity_ratio': 1.0,
            'spike_detected': False,
            'news_interpretation': "Nessuna news disponibile"
        }
    
    now = datetime.now()
    count_24h = 0
    count_7d = 0
    count_30d = 0
    
    for news in news_list:
        try:
            pub_date_str = news.get('published_at', '')
            if pub_date_str:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d']:
                    try:
                        pub_date = datetime.strptime(pub_date_str[:19], fmt[:19].replace('%z', ''))
                        break
                    except:
                        continue
                else:
                    continue
                
                days_ago = (now - pub_date).days
                
                if days_ago <= 1:
                    count_24h += 1
                if days_ago <= 7:
                    count_7d += 1
                if days_ago <= 30:
                    count_30d += 1
        except:
            continue
    
    avg_daily_30d = count_30d / 30 if count_30d > 0 else 0
    expected_7d = avg_daily_30d * 7
    velocity_ratio = count_7d / expected_7d if expected_7d > 0 else 1.0
    spike_detected = velocity_ratio > CONFIG['NEWS_SPIKE_THRESHOLD']
    
    if spike_detected:
        interpretation = "Picco anomalo di news - evento significativo in corso"
    elif velocity_ratio > 1.5:
        interpretation = "Attenzione mediatica sopra la media"
    elif velocity_ratio < 0.5:
        interpretation = "Copertura mediatica sotto la media"
    else:
        interpretation = "Copertura mediatica nella norma"
    
    return {
        'count_24h': count_24h,
        'count_7d': count_7d,
        'count_30d': count_30d,
        'avg_daily_30d': round(avg_daily_30d, 2),
        'velocity_ratio': round(velocity_ratio, 2),
        'spike_detected': spike_detected,
        'news_interpretation': interpretation
    }


def format_recent_news(news_list: List[Dict], max_items: int = 10) -> List[Dict]:
    """Formatta lista news per output."""
    if not news_list:
        return []
    
    formatted = []
    for news in news_list[:max_items]:
        sent_score = news.get('sentiment')
        if sent_score is not None:
            try:
                sent_val = float(sent_score)
                if sent_val > 0.2:
                    sent_label = "positive"
                elif sent_val < -0.2:
                    sent_label = "negative"
                else:
                    sent_label = "neutral"
            except:
                sent_label = None
                sent_score = None
        else:
            sent_label = None
        
        formatted.append({
            'title': news.get('title', '')[:200],
            'source': news.get('source', 'Unknown'),
            'published_at': news.get('published_at', ''),
            'sentiment_score': sent_score,
            'sentiment_label': sent_label,
            'url': news.get('url', '')
        })
    
    return formatted


# =============================================================================
# FUNZIONI COMPOSITE SCORING
# =============================================================================

def calculate_ema_component(ema_analysis: Dict) -> float:
    """Calcola componente EMA per technical score (0-100)."""
    distance_pct = ema_analysis['price_vs_ema_percent']
    
    if distance_pct > 10:
        position_score = 85
    elif distance_pct > 5:
        position_score = 75
    elif distance_pct > 2:
        position_score = 65
    elif distance_pct > 0:
        position_score = 55
    elif distance_pct > -2:
        position_score = 45
    elif distance_pct > -5:
        position_score = 35
    elif distance_pct > -10:
        position_score = 25
    else:
        position_score = 15
    
    slope_dir = ema_analysis['slope_direction']
    if slope_dir == 'accelerating_up':
        slope_adj = 10
    elif slope_dir == 'decelerating_up':
        slope_adj = 5
    elif slope_dir == 'decelerating_down':
        slope_adj = -5
    else:
        slope_adj = -10
    
    return max(0, min(100, position_score + slope_adj))


def calculate_rsi_component(rsi_analysis: Dict) -> float:
    """Calcola componente RSI per technical score (0-100)."""
    rsi_value = rsi_analysis['value']
    divergence = rsi_analysis['divergence_detected']
    
    if rsi_value >= 80:
        base_score = 75
    elif rsi_value >= 70:
        base_score = 70
    elif rsi_value >= 60:
        base_score = 62
    elif rsi_value >= 50:
        base_score = 55
    elif rsi_value >= 40:
        base_score = 45
    elif rsi_value >= 30:
        base_score = 38
    elif rsi_value >= 20:
        base_score = 30
    else:
        base_score = 25
    
    if divergence == 'bullish':
        base_score = min(100, base_score + 15)
    elif divergence == 'bearish':
        base_score = max(0, base_score - 15)
    
    return base_score


def calculate_momentum_component(momentum_analysis: Dict) -> float:
    """Calcola componente momentum per technical score (0-100)."""
    alignment = momentum_analysis['alignment']
    quality = momentum_analysis['momentum_quality']
    
    alignment_scores = {
        'accelerating_bullish': 90,
        'aligned_bullish': 75,
        'transitional': 50,
        'divergent': 45,
        'aligned_bearish': 25,
        'accelerating_bearish': 10
    }
    
    base_score = alignment_scores.get(alignment, 50)
    
    quality_adj = {
        'strong': 5,
        'moderate': 0,
        'weak': -5,
        'conflicting': -10
    }
    
    adj = quality_adj.get(quality, 0)
    return max(0, min(100, base_score + adj))


def calculate_sentiment_score(sentiment_analysis: Dict) -> float:
    """Calcola sentiment score complessivo (0-100)."""
    raw_norm = sentiment_analysis['current_sentiment']['normalized_0_100']
    
    momentum = sentiment_analysis['sentiment_dynamics']['momentum']
    momentum_scores = {
        'improving': 75,
        'recovering': 65,
        'stable': 50,
        'weakening': 35,
        'deteriorating': 25
    }
    momentum_score = momentum_scores.get(momentum, 50)
    
    news_data = sentiment_analysis['news_analysis']
    velocity = news_data['velocity_ratio']
    spike = news_data['spike_detected']
    
    if spike:
        if raw_norm > 60:
            velocity_score = 80
        elif raw_norm < 40:
            velocity_score = 20
        else:
            velocity_score = 50
    elif velocity > 1.5:
        velocity_score = 60
    elif velocity < 0.5:
        velocity_score = 40
    else:
        velocity_score = 50
    
    final_score = (raw_norm * 0.60) + (momentum_score * 0.25) + (velocity_score * 0.15)
    return round(final_score, 1)


def classify_signal(score: float) -> Tuple[str, str]:
    """Classifica segnale da score."""
    if score >= 80:
        return "strong_buy", "Convergenza positiva forte"
    elif score >= 65:
        return "buy", "Bias positivo"
    elif score >= 45:
        return "neutral", "Segnali misti"
    elif score >= 30:
        return "sell", "Bias negativo"
    else:
        return "strong_sell", "Convergenza negativa forte"


def calculate_confidence(
    momentum_alignment: str,
    volume_percentile: float,
    rsi_divergence: str,
    vol_regime: str,
    position_in_range: float
) -> Tuple[float, str]:
    """Calcola confidence score (0-1) e label."""
    confidence = 0.5
    
    if momentum_alignment in ['accelerating_bullish', 'accelerating_bearish']:
        confidence += 0.15
    elif momentum_alignment in ['aligned_bullish', 'aligned_bearish']:
        confidence += 0.10
    elif momentum_alignment == 'divergent':
        confidence -= 0.10
    
    if volume_percentile > 70:
        confidence += 0.15
    elif volume_percentile > 50:
        confidence += 0.08
    elif volume_percentile < 20:
        confidence -= 0.15
    elif volume_percentile < 35:
        confidence -= 0.08
    
    if rsi_divergence != 'none':
        confidence -= 0.05
    
    if vol_regime == 'normal':
        confidence += 0.05
    elif vol_regime == 'extreme':
        confidence -= 0.15
    elif vol_regime == 'high':
        confidence -= 0.05
    
    if position_in_range > 90 or position_in_range < 10:
        confidence -= 0.10
    elif position_in_range > 80 or position_in_range < 20:
        confidence -= 0.05
    
    confidence = max(0.1, min(0.95, confidence))
    
    if confidence >= 0.7:
        label = "high"
    elif confidence >= 0.45:
        label = "medium"
    else:
        label = "low"
    
    return round(confidence, 2), label


def generate_narrative(
    signal: str,
    confidence_label: str,
    bullish_factors: List[str],
    bearish_factors: List[str],
    asset_symbol: str
) -> str:
    """Genera narrativa testuale del segnale."""
    signal_text = {
        'strong_buy': 'fortemente rialzista',
        'buy': 'moderatamente rialzista',
        'neutral': 'neutrale',
        'sell': 'moderatamente ribassista',
        'strong_sell': 'fortemente ribassista'
    }
    
    conf_text = {
        'high': 'alta',
        'medium': 'media',
        'low': 'bassa'
    }
    
    narrative = f"L'analisi di {asset_symbol} indica un bias {signal_text.get(signal, 'neutrale')} "
    narrative += f"con confidence {conf_text.get(confidence_label, 'media')}. "
    
    if bullish_factors:
        narrative += f"Fattori positivi: {', '.join(bullish_factors[:3])}. "
    
    if bearish_factors:
        narrative += f"Fattori di rischio: {', '.join(bearish_factors[:3])}. "
    
    return narrative


# =============================================================================
# FUNZIONI GRAFICI PLOTLY
# =============================================================================

def create_gauge_chart(value: float, title: str, ranges: List[Dict] = None) -> go.Figure:
    """Crea gauge chart Plotly."""
    if ranges is None:
        ranges = [
            {'range': [0, 30], 'color': '#dc3545'},
            {'range': [30, 45], 'color': '#fd7e14'},
            {'range': [45, 65], 'color': '#ffc107'},
            {'range': [65, 80], 'color': '#28a745'},
            {'range': [80, 100], 'color': '#20c997'}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#333'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
            'bar': {'color': "#1a73e8"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': ranges,
            'threshold': {
                'line': {'color': "#333", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#333'}
    )
    
    return fig


def create_price_chart(df: pd.DataFrame, ema_period: int = 125) -> go.Figure:
    """Crea candlestick chart con EMA overlay."""
    plot_df = df.iloc[-252:].copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    ema_col = f'ema_{ema_period}'
    if ema_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[ema_col],
                mode='lines',
                name=f'EMA {ema_period}',
                line=dict(color='#ff9800', width=2)
            ),
            row=1, col=1
        )
    
    high_52w = plot_df['high'].max()
    low_52w = plot_df['low'].min()
    
    fig.add_hline(y=high_52w, line_dash="dash", line_color="#28a745",
                  annotation_text="52w High", row=1, col=1)
    fig.add_hline(y=low_52w, line_dash="dash", line_color="#dc3545",
                  annotation_text="52w Low", row=1, col=1)
    
    colors = ['#26a69a' if c >= o else '#ef5350'
              for c, o in zip(plot_df['close'], plot_df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    vol_sma = plot_df['volume'].rolling(20).mean()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=vol_sma,
            mode='lines',
            name='Vol SMA20',
            line=dict(color='#1a73e8', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
    
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Crea RSI panel con zone."""
    plot_df = df.iloc[-252:].copy()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['rsi_14'],
            mode='lines',
            name='RSI 14',
            line=dict(color='#7b1fa2', width=2)
        )
    )
    
    fig.add_hrect(y0=70, y1=100, fillcolor="#ffcdd2", opacity=0.3, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#c8e6c9", opacity=0.3, line_width=0)
    
    fig.add_hline(y=70, line_dash="dash", line_color="#dc3545", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", line_width=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#666", line_width=1)
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=50, r=50, t=10, b=30),
        yaxis=dict(range=[0, 100], title='RSI'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
    
    return fig


def create_momentum_chart(momentum_data: Dict) -> go.Figure:
    """Crea bar chart per ROC multi-timeframe."""
    periods = ['ROC 10d', 'ROC 21d', 'ROC 63d']
    values = [
        momentum_data['roc_10']['value'],
        momentum_data['roc_21']['value'],
        momentum_data['roc_63']['value']
    ]
    
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=periods,
            y=values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in values],
            textposition='outside'
        )
    )
    
    fig.add_hline(y=0, line_color="#333", line_width=1)
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=50, r=50, t=10, b=30),
        yaxis=dict(title='ROC %'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Crea chart volatilità HV20 vs HV60."""
    plot_df = df.iloc[-126:].copy()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['hv_20'] * 100,
            mode='lines',
            name='HV 20',
            line=dict(color='#1a73e8', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['hv_60'] * 100,
            mode='lines',
            name='HV 60',
            line=dict(color='#ff9800', width=2)
        )
    )
    
    fig.update_layout(
        height=200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=30, b=30),
        yaxis=dict(title='Volatility %'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
    
    return fig


def get_signal_color(signal: str) -> str:
    """Restituisce colore per segnale."""
    colors = {
        'strong_buy': '#20c997',
        'buy': '#28a745',
        'neutral': '#ffc107',
        'sell': '#fd7e14',
        'strong_sell': '#dc3545'
    }
    return colors.get(signal, '#6c757d')


def get_regime_color(regime: str) -> str:
    """Restituisce colore per regime volatilità."""
    colors = {
        'low': '#28a745',
        'normal': '#1a73e8',
        'high': '#fd7e14',
        'extreme': '#dc3545'
    }
    return colors.get(regime, '#6c757d')


# =============================================================================
# FUNZIONE PRINCIPALE DI ANALISI
# =============================================================================

def run_analysis(ticker: str, asset_info: Dict) -> Tuple[Dict, Dict, pd.DataFrame]:
    """Esegue l'analisi completa e restituisce tutti i dati."""
    
    # Fetch dati
    with st.spinner("📈 Recupero dati OHLCV..."):
        ohlcv_data = fetch_ohlcv_data(ticker, days=CONFIG['MIN_TRADING_DAYS'] + CONFIG['DATA_BUFFER_DAYS'])
    
    if ohlcv_data is None or len(ohlcv_data) < 50:
        st.error(f"❌ Dati OHLCV insufficienti per {ticker}. Minimo 50 giorni richiesti.")
        return None, None, None
    
    with st.spinner("💬 Recupero dati Sentiment..."):
        sentiment_data = fetch_sentiment_data(ticker, days=CONFIG['SENTIMENT_DAYS'])
    HAS_SENTIMENT = sentiment_data is not None
    
    with st.spinner("📰 Recupero News..."):
        news_data = fetch_news_data(ticker, limit=CONFIG['NEWS_LIMIT'])
    HAS_NEWS = news_data is not None and len(news_data) > 0
    
    # Calcolo indicatori
    ohlcv_data['ema_125'] = calculate_ema(ohlcv_data['close'], CONFIG['EMA_PERIOD'])
    ohlcv_data['rsi_14'] = calculate_rsi(ohlcv_data['close'], CONFIG['RSI_PERIOD'])
    ohlcv_data['atr_14'] = calculate_atr(ohlcv_data, CONFIG['ATR_PERIOD'])
    ohlcv_data['hv_20'] = calculate_historical_volatility(ohlcv_data['close'], CONFIG['HV_SHORT'])
    ohlcv_data['hv_60'] = calculate_historical_volatility(ohlcv_data['close'], CONFIG['HV_LONG'])
    
    for period in CONFIG['ROC_PERIODS']:
        ohlcv_data[f'roc_{period}'] = calculate_roc(ohlcv_data['close'], period)
    
    volume_percentile, ohlcv_data['volume_percentile'] = calculate_volume_percentile(
        ohlcv_data['volume'], CONFIG['VOLUME_LOOKBACK']
    )
    
    drawdown_df = calculate_drawdown(ohlcv_data['close'])
    ohlcv_data['running_max'] = drawdown_df['running_max']
    ohlcv_data['drawdown_pct'] = drawdown_df['drawdown_pct']
    
    # Valori correnti
    current_price = ohlcv_data['close'].iloc[-1]
    current_ema = ohlcv_data['ema_125'].iloc[-1]
    current_rsi = ohlcv_data['rsi_14'].iloc[-1]
    current_atr = ohlcv_data['atr_14'].iloc[-1]
    current_hv20 = ohlcv_data['hv_20'].iloc[-1]
    current_hv60 = ohlcv_data['hv_60'].iloc[-1]
    current_volume = ohlcv_data['volume'].iloc[-1]
    volume_sma_20 = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
    
    # EMA Analysis
    ema_distance_abs = current_price - current_ema
    ema_distance_pct = (ema_distance_abs / current_ema) * 100
    ema_position = "above" if current_price > current_ema else "below"
    
    ema_5d_ago = ohlcv_data['ema_125'].iloc[-6] if len(ohlcv_data) > 5 else ohlcv_data['ema_125'].iloc[0]
    ema_slope = (current_ema - ema_5d_ago) / 5
    ema_slope_normalized = (ema_slope / current_price) * 10000
    
    if len(ohlcv_data) > 10:
        ema_10d_ago = ohlcv_data['ema_125'].iloc[-11]
        prev_slope = (ema_5d_ago - ema_10d_ago) / 5
        slope_change = ema_slope - prev_slope
    else:
        slope_change = 0
    
    if ema_slope > 0:
        slope_direction = "accelerating_up" if slope_change > 0 else "decelerating_up"
    else:
        slope_direction = "accelerating_down" if slope_change < 0 else "decelerating_down"
    
    EMA_ANALYSIS = {
        'value': round(current_ema, 4),
        'price_vs_ema_absolute': round(ema_distance_abs, 4),
        'price_vs_ema_percent': round(ema_distance_pct, 2),
        'position': ema_position,
        'distance_interpretation': interpret_ema_distance(ema_distance_pct),
        'slope': round(ema_slope, 4),
        'slope_normalized_bps': round(ema_slope_normalized, 2),
        'slope_direction': slope_direction
    }
    
    # RSI Analysis
    rsi_zone, rsi_interpretation = interpret_rsi(current_rsi)
    RSI_ANALYSIS = {
        'value': round(current_rsi, 2),
        'zone': rsi_zone,
        'zone_interpretation': rsi_interpretation,
        'divergence_detected': 'none',
        'divergence_confidence': None
    }
    
    # Volume Analysis
    volume_ratio = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1.0
    
    if volume_percentile > 80:
        volume_confirmation = 1.2 + (volume_percentile - 80) / 100
    elif volume_percentile > 50:
        volume_confirmation = 1.0 + (volume_percentile - 50) / 150
    elif volume_percentile > 20:
        volume_confirmation = 0.85 + (volume_percentile - 20) / 200
    else:
        volume_confirmation = 0.7 + volume_percentile / 100
    
    VOLUME_ANALYSIS = {
        'current': int(current_volume),
        'sma_20': int(volume_sma_20),
        'ratio_vs_sma': round(volume_ratio, 2),
        'percentile_1y': round(volume_percentile, 1),
        'interpretation': interpret_volume_percentile(volume_percentile),
        'confirmation_factor': round(min(1.3, max(0.7, volume_confirmation)), 3)
    }
    
    # Volatility Analysis
    atr_pct_of_price = (current_atr / current_price) * 100
    available_atr_days = min(CONFIG['VOLUME_LOOKBACK'], len(ohlcv_data['atr_14'].dropna()))
    atr_window = ohlcv_data['atr_14'].dropna().iloc[-available_atr_days:]
    atr_percentile = stats.percentileofscore(atr_window.values, current_atr)
    
    hv_ratio = current_hv20 / current_hv60 if current_hv60 > 0 else 1.0
    available_hv_days = min(CONFIG['VOLUME_LOOKBACK'], len(ohlcv_data['hv_20'].dropna()))
    hv_window = ohlcv_data['hv_20'].dropna().iloc[-available_hv_days:]
    hv20_percentile = stats.percentileofscore(hv_window.values, current_hv20)
    
    vol_regime, vol_regime_interpretation = classify_volatility_regime(atr_percentile, hv20_percentile)
    
    if vol_regime == "extreme":
        volatility_adjustment = 0.8
    elif vol_regime == "high":
        volatility_adjustment = 0.9
    elif vol_regime == "low":
        volatility_adjustment = 1.1
    else:
        volatility_adjustment = 1.0
    
    VOLATILITY_ANALYSIS = {
        'atr_14': {
            'value': round(current_atr, 4),
            'as_percent_of_price': round(atr_pct_of_price, 2),
            'percentile_1y': round(atr_percentile, 1)
        },
        'historical_volatility': {
            'hv_20': round(current_hv20, 4),
            'hv_60': round(current_hv60, 4),
            'hv_ratio_20_60': round(hv_ratio, 3),
            'hv_20_percentile_1y': round(hv20_percentile, 1),
            'hv_interpretation': interpret_hv_ratio(hv_ratio)
        },
        'volatility_regime': vol_regime,
        'regime_interpretation': vol_regime_interpretation,
        'regime_impact_on_signals': (
            "Breakout signals più affidabili" if vol_regime in ['high', 'extreme']
            else "Mean reversion favorita" if vol_regime == 'low'
            else "Segnali standard"
        ),
        'volatility_adjustment_factor': volatility_adjustment
    }
    
    # Momentum Analysis
    roc_10 = ohlcv_data['roc_10'].iloc[-1]
    roc_21 = ohlcv_data['roc_21'].iloc[-1]
    roc_63 = ohlcv_data['roc_63'].iloc[-1]
    
    momentum_alignment, momentum_interpretation = classify_momentum_alignment(roc_10, roc_21, roc_63)
    
    rsi_divergence, divergence_confidence = detect_rsi_divergence(
        ohlcv_data['close'], ohlcv_data['rsi_14'], volume_percentile, lookback=14, tolerance=0.02
    )
    RSI_ANALYSIS['divergence_detected'] = rsi_divergence
    RSI_ANALYSIS['divergence_confidence'] = divergence_confidence
    
    momentum_quality = assess_momentum_quality(momentum_alignment, rsi_divergence, volume_percentile)
    
    MOMENTUM_ANALYSIS = {
        'roc_10': {'value': round(roc_10, 2), 'interpretation': interpret_single_roc(roc_10, 10)},
        'roc_21': {'value': round(roc_21, 2), 'interpretation': interpret_single_roc(roc_21, 21)},
        'roc_63': {'value': round(roc_63, 2), 'interpretation': interpret_single_roc(roc_63, 63)},
        'alignment': momentum_alignment,
        'alignment_interpretation': momentum_interpretation,
        'momentum_quality': momentum_quality
    }
    
    # Price Structure
    analysis_days = min(252, len(ohlcv_data))
    analysis_data = ohlcv_data.iloc[-analysis_days:]
    
    high_52w = analysis_data['high'].max()
    high_52w_date = analysis_data['high'].idxmax()
    low_52w = analysis_data['low'].min()
    low_52w_date = analysis_data['low'].idxmin()
    
    range_52w_abs = high_52w - low_52w
    range_52w_pct = (range_52w_abs / low_52w) * 100
    position_in_range = ((current_price - low_52w) / range_52w_abs) * 100 if range_52w_abs > 0 else 50
    range_zone, range_interpretation = interpret_range_position(position_in_range)
    
    distance_from_high_pct = ((current_price - high_52w) / high_52w) * 100
    distance_from_low_pct = ((current_price - low_52w) / low_52w) * 100
    days_since_high = (ohlcv_data.index[-1] - high_52w_date).days
    days_since_low = (ohlcv_data.index[-1] - low_52w_date).days
    
    current_drawdown = drawdown_df['drawdown_pct'].iloc[-1]
    max_drawdown_1y = analysis_data['close'].pipe(lambda x: calculate_drawdown(x)['drawdown_pct'].min())
    dd_series = drawdown_df['drawdown_pct'].iloc[-analysis_days:]
    dd_percentile = stats.percentileofscore(dd_series.values, current_drawdown)
    
    try:
        last_peak_idx = ohlcv_data.index.get_loc(
            ohlcv_data[ohlcv_data['close'] == ohlcv_data['running_max'].iloc[-1]].index[-1]
        )
        current_idx = len(ohlcv_data) - 1
        days_in_drawdown = current_idx - last_peak_idx if current_drawdown < 0 else 0
    except:
        days_in_drawdown = 0
    
    PRICE_STRUCTURE = {
        'range_52w': {
            'high': round(high_52w, 4),
            'high_date': high_52w_date.strftime('%Y-%m-%d'),
            'low': round(low_52w, 4),
            'low_date': low_52w_date.strftime('%Y-%m-%d'),
            'range_absolute': round(range_52w_abs, 4),
            'range_percent': round(range_52w_pct, 2)
        },
        'position_in_range': {
            'value_percent': round(position_in_range, 1),
            'zone': range_zone,
            'interpretation': range_interpretation
        },
        'distance_from_levels': {
            'from_52w_high_pct': round(distance_from_high_pct, 2),
            'from_52w_low_pct': round(distance_from_low_pct, 2),
            'days_since_52w_high': days_since_high,
            'days_since_52w_low': days_since_low
        },
        'drawdown': {
            'current_drawdown_pct': round(current_drawdown, 2),
            'max_drawdown_1y_pct': round(max_drawdown_1y, 2),
            'drawdown_percentile': round(dd_percentile, 1),
            'days_in_drawdown': days_in_drawdown,
            'drawdown_interpretation': interpret_drawdown(current_drawdown)
        }
    }
    
    # Current Price
    price_1w_ago = ohlcv_data['close'].iloc[-6] if len(ohlcv_data) > 5 else ohlcv_data['close'].iloc[0]
    price_1m_ago = ohlcv_data['close'].iloc[-22] if len(ohlcv_data) > 21 else ohlcv_data['close'].iloc[0]
    price_3m_ago = ohlcv_data['close'].iloc[-64] if len(ohlcv_data) > 63 else ohlcv_data['close'].iloc[0]
    
    change_1w_pct = ((current_price - price_1w_ago) / price_1w_ago) * 100
    change_1m_pct = ((current_price - price_1m_ago) / price_1m_ago) * 100
    change_3m_pct = ((current_price - price_3m_ago) / price_3m_ago) * 100
    
    previous_close = ohlcv_data['close'].iloc[-2]
    change_abs = current_price - previous_close
    change_pct = ((current_price - previous_close) / previous_close) * 100
    
    CURRENT_PRICE = {
        'value': round(current_price, 4),
        'previous_close': round(previous_close, 4),
        'change_absolute': round(change_abs, 4),
        'change_percent': round(change_pct, 2),
        'change_1w_pct': round(change_1w_pct, 2),
        'change_1m_pct': round(change_1m_pct, 2),
        'change_3m_pct': round(change_3m_pct, 2),
        'last_updated': ohlcv_data.index[-1].strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    
    # Sentiment Processing
    SENTIMENT_ANALYSIS = {
        'current_sentiment': {
            'score': None,
            'normalized_0_100': 50.0,
            'label': 'neutral'
        },
        'sentiment_dynamics': {
            'score_7d_ago': None,
            'score_30d_ago': None,
            'change_7d': 0,
            'change_30d': 0,
            'momentum': 'stable',
            'momentum_interpretation': 'Sentiment stabile'
        },
        'news_analysis': {
            'count_24h': 0,
            'count_7d': 0,
            'count_30d': 0,
            'avg_daily_30d': 0,
            'velocity_ratio': 1.0,
            'spike_detected': False,
            'news_interpretation': 'Copertura mediatica nella norma'
        },
        'buzz_score': None
    }
    
    RECENT_NEWS = []
    
    if HAS_SENTIMENT and isinstance(sentiment_data, dict):
        raw_score = sentiment_data.get('sentiment',
                    sentiment_data.get('score',
                    sentiment_data.get('sentiment_score',
                    sentiment_data.get('normalized', None))))
        
        buzz = sentiment_data.get('buzz', sentiment_data.get('buzz_score', None))
        
        if raw_score is not None:
            try:
                raw_score = float(raw_score)
                normalized_score = normalize_sentiment_score(raw_score)
                sentiment_label = classify_sentiment_label(normalized_score)
                
                SENTIMENT_ANALYSIS['current_sentiment'] = {
                    'score': round(raw_score, 3),
                    'normalized_0_100': normalized_score,
                    'label': sentiment_label
                }
            except:
                pass
        
        if buzz is not None:
            try:
                SENTIMENT_ANALYSIS['buzz_score'] = float(buzz)
            except:
                pass
    
    if HAS_NEWS:
        news_metrics = analyze_news_velocity(news_data)
        SENTIMENT_ANALYSIS['news_analysis'] = news_metrics
        RECENT_NEWS = format_recent_news(news_data, max_items=10)
    
    # Composite Score
    ema_component = calculate_ema_component(EMA_ANALYSIS)
    rsi_component = calculate_rsi_component(RSI_ANALYSIS)
    momentum_component = calculate_momentum_component(MOMENTUM_ANALYSIS)
    
    technical_score = (
        ema_component * CONFIG['W_EMA'] +
        rsi_component * CONFIG['W_RSI'] +
        momentum_component * CONFIG['W_MOMENTUM']
    )
    
    sentiment_score = calculate_sentiment_score(SENTIMENT_ANALYSIS)
    
    raw_composite = (
        technical_score * CONFIG['W_TECHNICAL'] +
        sentiment_score * CONFIG['W_SENTIMENT']
    )
    
    volume_factor = VOLUME_ANALYSIS['confirmation_factor']
    volatility_factor = VOLATILITY_ANALYSIS['volatility_adjustment_factor']
    
    adjusted_composite = raw_composite * volume_factor * volatility_factor
    final_score = max(0, min(100, adjusted_composite))
    
    signal, signal_description = classify_signal(final_score)
    
    confidence, confidence_label = calculate_confidence(
        MOMENTUM_ANALYSIS['alignment'],
        VOLUME_ANALYSIS['percentile_1y'],
        RSI_ANALYSIS['divergence_detected'],
        VOLATILITY_ANALYSIS['volatility_regime'],
        PRICE_STRUCTURE['position_in_range']['value_percent']
    )
    
    # Key Factors
    bullish_factors = []
    bearish_factors = []
    neutral_factors = []
    
    if EMA_ANALYSIS['position'] == 'above':
        bullish_factors.append(f"Prezzo sopra EMA125 ({EMA_ANALYSIS['price_vs_ema_percent']:+.1f}%)")
    else:
        bearish_factors.append(f"Prezzo sotto EMA125 ({EMA_ANALYSIS['price_vs_ema_percent']:+.1f}%)")
    
    if RSI_ANALYSIS['zone'] in ['bullish', 'overbought']:
        bullish_factors.append(f"RSI in zona {RSI_ANALYSIS['zone']} ({RSI_ANALYSIS['value']:.0f})")
    elif RSI_ANALYSIS['zone'] in ['bearish', 'oversold']:
        bearish_factors.append(f"RSI in zona {RSI_ANALYSIS['zone']} ({RSI_ANALYSIS['value']:.0f})")
    
    if RSI_ANALYSIS['divergence_detected'] == 'bullish':
        bullish_factors.append("Divergenza RSI bullish rilevata")
    elif RSI_ANALYSIS['divergence_detected'] == 'bearish':
        bearish_factors.append("Divergenza RSI bearish rilevata")
    
    if 'bullish' in MOMENTUM_ANALYSIS['alignment']:
        bullish_factors.append(f"Momentum {MOMENTUM_ANALYSIS['alignment'].replace('_', ' ')}")
    elif 'bearish' in MOMENTUM_ANALYSIS['alignment']:
        bearish_factors.append(f"Momentum {MOMENTUM_ANALYSIS['alignment'].replace('_', ' ')}")
    else:
        neutral_factors.append("Momentum in transizione")
    
    if VOLUME_ANALYSIS['percentile_1y'] > 70:
        bullish_factors.append(f"Volume elevato (P{VOLUME_ANALYSIS['percentile_1y']:.0f})")
    elif VOLUME_ANALYSIS['percentile_1y'] < 30:
        bearish_factors.append(f"Volume basso (P{VOLUME_ANALYSIS['percentile_1y']:.0f})")
    
    if VOLATILITY_ANALYSIS['volatility_regime'] == 'extreme':
        bearish_factors.append("Volatilità estrema - rischio elevato")
    elif VOLATILITY_ANALYSIS['volatility_regime'] == 'low':
        neutral_factors.append("Bassa volatilità - breakout meno affidabili")
    
    if SENTIMENT_ANALYSIS['current_sentiment']['normalized_0_100'] > 65:
        bullish_factors.append(f"Sentiment positivo ({SENTIMENT_ANALYSIS['current_sentiment']['label']})")
    elif SENTIMENT_ANALYSIS['current_sentiment']['normalized_0_100'] < 35:
        bearish_factors.append(f"Sentiment negativo ({SENTIMENT_ANALYSIS['current_sentiment']['label']})")
    
    if SENTIMENT_ANALYSIS['news_analysis']['spike_detected']:
        neutral_factors.append("Spike anomalo di news - attenzione")
    
    narrative = generate_narrative(signal, confidence_label, bullish_factors, bearish_factors, ticker)
    
    COMPOSITE_ANALYSIS = {
        'scores': {
            'technical_score': round(technical_score, 1),
            'sentiment_score': round(sentiment_score, 1),
            'raw_composite': round(raw_composite, 1),
            'volume_adjustment': round(volume_factor, 3),
            'volatility_adjustment': round(volatility_factor, 2),
            'final_score': round(final_score, 1)
        },
        'signal': {
            'value': signal,
            'confidence': confidence,
            'confidence_label': confidence_label
        },
        'score_breakdown': {
            'technical_contribution': round(technical_score * CONFIG['W_TECHNICAL'], 1),
            'sentiment_contribution': round(sentiment_score * CONFIG['W_SENTIMENT'], 1),
            'ema_component': round(ema_component, 1),
            'rsi_component': round(rsi_component, 1),
            'momentum_component': round(momentum_component, 1)
        },
        'key_factors': {
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'neutral_factors': neutral_factors
        },
        'narrative_summary': narrative
    }
    
    # Build JSON Export
    signal_text = COMPOSITE_ANALYSIS['signal']['value'].replace('_', ' ').upper()
    confidence_pct = f"{COMPOSITE_ANALYSIS['signal']['confidence']:.0%}"
    
    json_export = {
        'metadata': {
            'asset_symbol': ticker,
            'asset_name': asset_info.get('symbol', ticker),
            'asset_type': asset_info.get('type', 'equity'),
            'exchange': asset_info.get('exchange', 'Unknown'),
            'currency': asset_info.get('currency', 'USD'),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'data_range': {
                'start_date': ohlcv_data.index[0].strftime('%Y-%m-%d'),
                'end_date': ohlcv_data.index[-1].strftime('%Y-%m-%d'),
                'trading_days': len(ohlcv_data)
            },
            'generator': 'Kriterion Quant Sentiment Analyzer v2.0',
            'data_quality': {
                'has_sentiment': HAS_SENTIMENT,
                'has_news': HAS_NEWS,
                'ohlcv_complete': len(ohlcv_data) >= CONFIG['MIN_TRADING_DAYS']
            }
        },
        'current_price': CURRENT_PRICE,
        'technical_indicators': {
            'ema_125': EMA_ANALYSIS,
            'rsi_14': RSI_ANALYSIS,
            'volume': VOLUME_ANALYSIS
        },
        'volatility_analysis': VOLATILITY_ANALYSIS,
        'momentum_multitimeframe': MOMENTUM_ANALYSIS,
        'price_structure': PRICE_STRUCTURE,
        'sentiment_analysis': SENTIMENT_ANALYSIS,
        'recent_news': RECENT_NEWS,
        'composite_analysis': COMPOSITE_ANALYSIS,
        'llm_analysis_prompt': {
            'context': (
                f"Analisi di {asset_info.get('symbol', ticker)} ({ticker}) generata il {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
                f"Prezzo corrente {CURRENT_PRICE['value']:.4f} {asset_info.get('currency', 'USD')} ({CURRENT_PRICE['change_percent']:+.2f}%). "
                f"Signal: {signal_text} con confidence {confidence_pct}. "
                f"Composite Score: {COMPOSITE_ANALYSIS['scores']['final_score']:.1f}/100."
            ),
            'key_questions': [
                f"Quali sono i principali driver del segnale {signal_text}?",
                "Esistono divergenze tra sentiment e technical analysis?",
                "Il volume conferma o smentisce il movimento recente?",
                "Il regime di volatilità attuale come influenza l'affidabilità del segnale?",
                "Quali sono i rischi principali della posizione suggerita?",
                "Come si posiziona l'asset rispetto al range 52 settimane?",
                "Il momentum multi-timeframe è allineato o divergente?"
            ],
            'suggested_focus_areas': [
                f"Analisi della divergenza RSI: {RSI_ANALYSIS['divergence_detected']}",
                f"Valutazione del momentum multi-timeframe: {MOMENTUM_ANALYSIS['alignment']}",
                f"Contestualizzazione nel range 52 settimane: {PRICE_STRUCTURE['position_in_range']['zone']}",
                f"Regime volatilità: {VOLATILITY_ANALYSIS['volatility_regime']} - {VOLATILITY_ANALYSIS['regime_interpretation']}",
                f"Trend del sentiment: {SENTIMENT_ANALYSIS['sentiment_dynamics']['momentum']}"
            ],
            'analysis_request_template': (
                f"Analizza i dati di {ticker} considerando:\n"
                f"1. Il segnale composito è {signal_text} con score {COMPOSITE_ANALYSIS['scores']['final_score']:.1f}/100\n"
                f"2. Technical Score: {COMPOSITE_ANALYSIS['scores']['technical_score']:.1f} | Sentiment Score: {COMPOSITE_ANALYSIS['scores']['sentiment_score']:.1f}\n"
                f"3. Momentum: {MOMENTUM_ANALYSIS['alignment']} | Volatilità: {VOLATILITY_ANALYSIS['volatility_regime']}\n"
                f"4. Position in 52w range: {PRICE_STRUCTURE['position_in_range']['value_percent']:.1f}%\n"
                f"5. Drawdown corrente: {PRICE_STRUCTURE['drawdown']['current_drawdown_pct']:.2f}%\n\n"
                f"Fornisci un'analisi dettagliata con raccomandazioni operative e gestione del rischio."
            )
        }
    }
    
    # Dati per dashboard
    dashboard_data = {
        'TICKER': ticker,
        'ASSET_INFO': asset_info,
        'CONFIG': CONFIG,
        'CURRENT_PRICE': CURRENT_PRICE,
        'EMA_ANALYSIS': EMA_ANALYSIS,
        'RSI_ANALYSIS': RSI_ANALYSIS,
        'VOLUME_ANALYSIS': VOLUME_ANALYSIS,
        'VOLATILITY_ANALYSIS': VOLATILITY_ANALYSIS,
        'MOMENTUM_ANALYSIS': MOMENTUM_ANALYSIS,
        'PRICE_STRUCTURE': PRICE_STRUCTURE,
        'SENTIMENT_ANALYSIS': SENTIMENT_ANALYSIS,
        'RECENT_NEWS': RECENT_NEWS,
        'COMPOSITE_ANALYSIS': COMPOSITE_ANALYSIS,
        'HAS_SENTIMENT': HAS_SENTIMENT,
        'HAS_NEWS': HAS_NEWS
    }
    
    return json_export, dashboard_data, ohlcv_data


# =============================================================================
# INTERFACCIA STREAMLIT
# =============================================================================

def main():
    # Sidebar
    st.sidebar.image("https://raw.githubusercontent.com/kriterion-quant/kriterion-quant.github.io/main/assets/logo.png", width=200)
    st.sidebar.title("📊 Sentiment Analysis")
    st.sidebar.markdown("---")
    
    # Verifica API Key
    if not EODHD_API_KEY:
        st.sidebar.error("⚠️ API Key EODHD non configurata!")
        st.sidebar.info("Configura EODHD_API_KEY nei Secrets di Streamlit")
        st.stop()
    else:
        st.sidebar.success("✅ API Key configurata")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Formati supportati")
    st.sidebar.markdown("""
    - **Equity US:** AAPL.US, MSFT.US
    - **Equity EU:** ENI.MI, SAP.F
    - **Crypto:** BTC-USD.CC, ETH-USD.CC
    - **ETF:** SPY.US, QQQ.US
    - **Index:** GSPC.INDX, DJI.INDX
    """)
    
    st.sidebar.markdown("---")
    
    # Input ticker
    ticker_input = st.sidebar.text_input(
        "🔍 Inserisci Ticker",
        value="AAPL.US",
        help="Formato: SYMBOL.EXCHANGE"
    ).strip().upper()
    
    analyze_button = st.sidebar.button("🚀 Analizza", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    
    # Placeholder per download JSON
    json_download_placeholder = st.sidebar.empty()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Info")
    st.sidebar.markdown("""
    **Kriterion Quant v2.0**  
    Sentiment & Technical Analysis  
    Data: [EODHD](https://eodhd.com)
    """)
    
    # Main content
    st.title("🎯 Financial Sentiment Analysis Dashboard")
    st.markdown("**Kriterion Quant** - Analisi tecnica e sentiment integrata")
    
    if analyze_button:
        # Validazione ticker
        is_valid, error_msg, asset_info = validate_ticker(ticker_input)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.stop()
        
        st.info(f"📊 Analisi in corso per **{ticker_input}** ({asset_info['name']})...")
        
        # Esegui analisi
        json_export, dashboard_data, ohlcv_data = run_analysis(ticker_input, asset_info)
        
        if json_export is None:
            st.stop()
        
        # Pulsante download JSON nella sidebar
        json_string = json.dumps(json_export, indent=2, ensure_ascii=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        ticker_clean = ticker_input.replace('.', '_').replace('-', '_')
        json_filename = f"{ticker_clean}_{timestamp}_analysis_data.json"
        
        json_download_placeholder.download_button(
            label="📥 Download JSON",
            data=json_string,
            file_name=json_filename,
            mime="application/json",
            use_container_width=True
        )
        
        # =====================================================================
        # DASHBOARD
        # =====================================================================
        
        d = dashboard_data  # Shortcut
        
        # Header
        change_color = "green" if d['CURRENT_PRICE']['change_percent'] >= 0 else "red"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); 
                    color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h2 style="margin: 0;">{d['ASSET_INFO'].get('symbol', d['TICKER'])} - {d['ASSET_INFO'].get('name', 'Asset')}</h2>
            <p style="opacity: 0.9; margin: 5px 0;">Sentiment & Technical Analysis Dashboard</p>
            <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px;">
                <span style="background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px;">
                    📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </span>
                <span style="background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px;">
                    💰 {d['CURRENT_PRICE']['value']:.4f} {d['ASSET_INFO'].get('currency', 'USD')}
                </span>
                <span style="background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px; color: {'#a5d6a7' if d['CURRENT_PRICE']['change_percent'] >= 0 else '#ef9a9a'};">
                    {d['CURRENT_PRICE']['change_percent']:+.2f}% (1D)
                </span>
                <span style="background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px;">
                    📊 {d['ASSET_INFO'].get('type', 'equity').upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Score Cards Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Composite Score")
            fig_gauge = create_gauge_chart(d['COMPOSITE_ANALYSIS']['scores']['final_score'], '')
            st.plotly_chart(fig_gauge, use_container_width=True)
            signal_color = get_signal_color(d['COMPOSITE_ANALYSIS']['signal']['value'])
            st.markdown(f"""
            <div style="text-align: center;">
                <span style="background-color: {signal_color}; color: white; padding: 8px 20px; 
                            border-radius: 25px; font-weight: 600; text-transform: uppercase;">
                    {d['COMPOSITE_ANALYSIS']['signal']['value'].replace('_', ' ')}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Sentiment Score")
            fig_sentiment = create_gauge_chart(
                d['SENTIMENT_ANALYSIS']['current_sentiment']['normalized_0_100'],
                '',
                [
                    {'range': [0, 20], 'color': '#dc3545'},
                    {'range': [20, 35], 'color': '#fd7e14'},
                    {'range': [35, 50], 'color': '#ffc107'},
                    {'range': [50, 65], 'color': '#90caf9'},
                    {'range': [65, 80], 'color': '#28a745'},
                    {'range': [80, 100], 'color': '#20c997'}
                ]
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.markdown(f"<p style='text-align: center;'>{d['SENTIMENT_ANALYSIS']['current_sentiment']['label'].replace('_', ' ').title()}</p>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### Technical Bias")
            st.metric(
                label="Score",
                value=f"{d['COMPOSITE_ANALYSIS']['scores']['technical_score']:.0f}/100"
            )
            st.caption(f"EMA: {d['EMA_ANALYSIS']['position'].upper()}")
            st.caption(f"RSI: {d['RSI_ANALYSIS']['zone'].replace('_', ' ').title()}")
            st.caption(f"Momentum: {d['MOMENTUM_ANALYSIS']['alignment'].replace('_', ' ').title()}")
        
        with col4:
            st.markdown("#### Volatility Regime")
            regime_color = get_regime_color(d['VOLATILITY_ANALYSIS']['volatility_regime'])
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <span style="background-color: {regime_color}; color: white; padding: 6px 15px; 
                            border-radius: 20px; font-weight: 500; text-transform: uppercase;">
                    {d['VOLATILITY_ANALYSIS']['volatility_regime']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"ATR%: {d['VOLATILITY_ANALYSIS']['atr_14']['as_percent_of_price']:.2f}%")
            st.caption(f"HV20/60: {d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_ratio_20_60']:.2f}")
        
        st.markdown("---")
        
        # 52W Range Bar
        st.markdown("### 📏 52 Week Range Position")
        range_position = d['PRICE_STRUCTURE']['position_in_range']['value_percent']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Low:** {d['PRICE_STRUCTURE']['range_52w']['low']:.4f}")
        with col2:
            st.caption(f"**Current:** {d['CURRENT_PRICE']['value']:.4f}")
        with col3:
            st.caption(f"**High:** {d['PRICE_STRUCTURE']['range_52w']['high']:.4f}")
        
        st.progress(range_position / 100)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Position", f"{range_position:.1f}%")
        with col2:
            st.metric("Current DD", f"{d['PRICE_STRUCTURE']['drawdown']['current_drawdown_pct']:.2f}%")
        with col3:
            st.metric("Days from High", d['PRICE_STRUCTURE']['distance_from_levels']['days_since_52w_high'])
        with col4:
            st.metric("Max DD 1Y", f"{d['PRICE_STRUCTURE']['drawdown']['max_drawdown_1y_pct']:.2f}%")
        
        st.markdown("---")
        
        # Price Chart
        st.markdown(f"### 📈 Price Action & EMA{d['CONFIG']['EMA_PERIOD']}")
        fig_price = create_price_chart(ohlcv_data, d['CONFIG']['EMA_PERIOD'])
        st.plotly_chart(fig_price, use_container_width=True)
        
        # RSI & Momentum Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📉 RSI {d['CONFIG']['RSI_PERIOD']}")
            fig_rsi = create_rsi_chart(ohlcv_data)
            st.plotly_chart(fig_rsi, use_container_width=True)
            st.caption(f"**Current:** {d['RSI_ANALYSIS']['value']:.1f} | **Zone:** {d['RSI_ANALYSIS']['zone'].replace('_', ' ').title()} | **Divergence:** {d['RSI_ANALYSIS']['divergence_detected'].title()}")
        
        with col2:
            st.markdown("### 🚀 Multi-Timeframe Momentum")
            fig_momentum = create_momentum_chart(d['MOMENTUM_ANALYSIS'])
            st.plotly_chart(fig_momentum, use_container_width=True)
            st.caption(f"**Alignment:** {d['MOMENTUM_ANALYSIS']['alignment'].replace('_', ' ').title()} | **Quality:** {d['MOMENTUM_ANALYSIS']['momentum_quality'].title()}")
        
        # Volatility Chart
        st.markdown("### 📊 Historical Volatility (HV20 vs HV60)")
        fig_vol = create_volatility_chart(ohlcv_data)
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption(f"**HV20:** {d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_20']*100:.1f}% | **HV60:** {d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_60']*100:.1f}% | **Trend:** {d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_interpretation']}")
        
        st.markdown("---")
        
        # Analysis Summary
        st.markdown("### 📝 Analysis Summary")
        st.info(d['COMPOSITE_ANALYSIS']['narrative_summary'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Bullish Factors**")
            if d['COMPOSITE_ANALYSIS']['key_factors']['bullish_factors']:
                for f in d['COMPOSITE_ANALYSIS']['key_factors']['bullish_factors']:
                    st.success(f)
            else:
                st.caption("Nessun fattore bullish rilevante")
        
        with col2:
            st.markdown("**⚠️ Bearish Factors**")
            if d['COMPOSITE_ANALYSIS']['key_factors']['bearish_factors']:
                for f in d['COMPOSITE_ANALYSIS']['key_factors']['bearish_factors']:
                    st.error(f)
            else:
                st.caption("Nessun fattore bearish rilevante")
        
        with col3:
            st.markdown("**➖ Neutral Factors**")
            if d['COMPOSITE_ANALYSIS']['key_factors']['neutral_factors']:
                for f in d['COMPOSITE_ANALYSIS']['key_factors']['neutral_factors']:
                    st.warning(f)
            else:
                st.caption("Nessun fattore neutrale")
        
        st.markdown("---")
        
        # Confidence Info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal Confidence", f"{d['COMPOSITE_ANALYSIS']['signal']['confidence']:.0%}", d['COMPOSITE_ANALYSIS']['signal']['confidence_label'].upper())
        with col2:
            st.metric("Volume Confirmation", f"{d['VOLUME_ANALYSIS']['confirmation_factor']:.2f}x", f"P{d['VOLUME_ANALYSIS']['percentile_1y']:.0f}")
        with col3:
            st.metric("Volatility Adjustment", f"{d['VOLATILITY_ANALYSIS']['volatility_adjustment_factor']:.2f}x", d['VOLATILITY_ANALYSIS']['regime_impact_on_signals'])
        
        st.markdown("---")
        
        # News Section
        st.markdown("### 📰 Recent News")
        
        news_col1, news_col2, news_col3 = st.columns(3)
        with news_col1:
            st.metric("News 24h", d['SENTIMENT_ANALYSIS']['news_analysis']['count_24h'])
        with news_col2:
            st.metric("News 7d", d['SENTIMENT_ANALYSIS']['news_analysis']['count_7d'])
        with news_col3:
            velocity = d['SENTIMENT_ANALYSIS']['news_analysis']['velocity_ratio']
            spike = "🔴 SPIKE" if d['SENTIMENT_ANALYSIS']['news_analysis']['spike_detected'] else ""
            st.metric("Velocity", f"{velocity:.2f}x", spike)
        
        if d['RECENT_NEWS']:
            for news in d['RECENT_NEWS'][:10]:
                with st.expander(news['title'][:100] + "..." if len(news['title']) > 100 else news['title']):
                    st.caption(f"**Source:** {news['source']} | **Date:** {news['published_at'][:10] if news['published_at'] else 'N/A'}")
                    if news['sentiment_label']:
                        sent_color = {'positive': '🟢', 'negative': '🔴', 'neutral': '⚪'}.get(news['sentiment_label'], '⚪')
                        st.caption(f"**Sentiment:** {sent_color} {news['sentiment_label'].title()}")
                    if news['url']:
                        st.markdown(f"[🔗 Link all'articolo]({news['url']})")
        else:
            st.info("Nessuna news disponibile per questo asset.")
        
        st.markdown("---")
        
        # Detailed Metrics Table
        st.markdown("### 📋 Detailed Metrics")
        
        with st.expander("📊 Visualizza tutte le metriche", expanded=False):
            
            st.markdown("#### Price Data")
            price_data = {
                'Metric': ['Current Price', 'Change (1D)', 'Change (1W)', 'Change (1M)', 'Change (3M)'],
                'Value': [
                    f"{d['CURRENT_PRICE']['value']:.4f} {d['ASSET_INFO'].get('currency', 'USD')}",
                    f"{d['CURRENT_PRICE']['change_percent']:+.2f}%",
                    f"{d['CURRENT_PRICE']['change_1w_pct']:+.2f}%",
                    f"{d['CURRENT_PRICE']['change_1m_pct']:+.2f}%",
                    f"{d['CURRENT_PRICE']['change_3m_pct']:+.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(price_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### Technical Indicators")
            tech_data = {
                'Metric': [
                    f"EMA {d['CONFIG']['EMA_PERIOD']}", 'EMA Slope', f"RSI {d['CONFIG']['RSI_PERIOD']}", 'RSI Divergence',
                    'ROC 10d', 'ROC 21d', 'ROC 63d', 'Momentum Alignment'
                ],
                'Value': [
                    f"{d['EMA_ANALYSIS']['value']:.4f}",
                    f"{d['EMA_ANALYSIS']['slope_normalized_bps']:.2f} bps",
                    f"{d['RSI_ANALYSIS']['value']:.1f}",
                    d['RSI_ANALYSIS']['divergence_detected'].title(),
                    f"{d['MOMENTUM_ANALYSIS']['roc_10']['value']:+.2f}%",
                    f"{d['MOMENTUM_ANALYSIS']['roc_21']['value']:+.2f}%",
                    f"{d['MOMENTUM_ANALYSIS']['roc_63']['value']:+.2f}%",
                    d['MOMENTUM_ANALYSIS']['alignment'].replace('_', ' ').title()
                ],
                'Interpretation': [
                    f"{d['EMA_ANALYSIS']['position'].upper()} ({d['EMA_ANALYSIS']['price_vs_ema_percent']:+.2f}%)",
                    d['EMA_ANALYSIS']['slope_direction'].replace('_', ' ').title(),
                    d['RSI_ANALYSIS']['zone'].replace('_', ' ').title(),
                    d['RSI_ANALYSIS']['divergence_confidence'] or 'N/A',
                    d['MOMENTUM_ANALYSIS']['roc_10']['interpretation'],
                    d['MOMENTUM_ANALYSIS']['roc_21']['interpretation'],
                    d['MOMENTUM_ANALYSIS']['roc_63']['interpretation'],
                    d['MOMENTUM_ANALYSIS']['momentum_quality'].title()
                ]
            }
            st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### Volatility")
            vol_data = {
                'Metric': [f"ATR {d['CONFIG']['ATR_PERIOD']}", 'ATR Percentile', 'HV 20', 'HV 60', 'HV Ratio', 'Volatility Regime'],
                'Value': [
                    f"{d['VOLATILITY_ANALYSIS']['atr_14']['value']:.4f}",
                    f"{d['VOLATILITY_ANALYSIS']['atr_14']['percentile_1y']:.1f}%",
                    f"{d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_20']*100:.1f}%",
                    f"{d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_60']*100:.1f}%",
                    f"{d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_ratio_20_60']:.3f}",
                    d['VOLATILITY_ANALYSIS']['volatility_regime'].upper()
                ],
                'Interpretation': [
                    f"{d['VOLATILITY_ANALYSIS']['atr_14']['as_percent_of_price']:.2f}% of price",
                    '-',
                    'Annualized',
                    'Annualized',
                    d['VOLATILITY_ANALYSIS']['historical_volatility']['hv_interpretation'],
                    d['VOLATILITY_ANALYSIS']['regime_interpretation']
                ]
            }
            st.dataframe(pd.DataFrame(vol_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### Volume")
            volume_data = {
                'Metric': ['Current Volume', 'Volume SMA 20', 'Volume Percentile'],
                'Value': [
                    f"{d['VOLUME_ANALYSIS']['current']:,}",
                    f"{d['VOLUME_ANALYSIS']['sma_20']:,}",
                    f"{d['VOLUME_ANALYSIS']['percentile_1y']:.1f}%"
                ],
                'Interpretation': [
                    '-',
                    f"{d['VOLUME_ANALYSIS']['ratio_vs_sma']:.2f}x",
                    d['VOLUME_ANALYSIS']['interpretation']
                ]
            }
            st.dataframe(pd.DataFrame(volume_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### Composite Score")
            comp_data = {
                'Metric': ['Technical Score', 'Sentiment Score', 'Raw Composite', 'Final Score', 'Signal'],
                'Value': [
                    f"{d['COMPOSITE_ANALYSIS']['scores']['technical_score']:.1f}/100",
                    f"{d['COMPOSITE_ANALYSIS']['scores']['sentiment_score']:.1f}/100",
                    f"{d['COMPOSITE_ANALYSIS']['scores']['raw_composite']:.1f}/100",
                    f"{d['COMPOSITE_ANALYSIS']['scores']['final_score']:.1f}/100",
                    d['COMPOSITE_ANALYSIS']['signal']['value'].replace('_', ' ').upper()
                ],
                'Details': [
                    f"Weight: {d['CONFIG']['W_TECHNICAL']}",
                    f"Weight: {d['CONFIG']['W_SENTIMENT']}",
                    '-',
                    'After adjustments',
                    f"Confidence: {d['COMPOSITE_ANALYSIS']['signal']['confidence']:.0%}"
                ]
            }
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            <p>Generated by <strong>Kriterion Quant</strong> - Financial Sentiment Analysis Dashboard v2.0</p>
            <p>Data source: <a href="https://eodhd.com" target="_blank">EODHD</a></p>
            <p style="margin-top: 10px; font-size: 0.8rem; color: #999;">
                ⚠️ This analysis is for informational purposes only and does not constitute financial advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Pagina iniziale
        st.markdown("""
        ### 👋 Benvenuto!
        
        Questa dashboard ti permette di analizzare qualsiasi strumento finanziario combinando:
        
        - 📈 **Analisi Tecnica** (EMA, RSI, Momentum, Volatilità)
        - 💬 **Analisi Sentiment** (News, Social Media)
        - 🎯 **Score Composito** con segnale operativo
        
        **Come iniziare:**
        1. Inserisci un ticker nella sidebar (es. AAPL.US)
        2. Clicca su "Analizza"
        3. Esplora i risultati e scarica il JSON per analisi con LLM
        
        ---
        
        *Powered by Kriterion Quant | Data by EODHD*
        """)


if __name__ == "__main__":
    main()
