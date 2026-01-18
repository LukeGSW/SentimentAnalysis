# 📊 Kriterion Quant - Financial Sentiment Analysis Dashboard

Dashboard interattiva per l'analisi tecnica e sentiment di strumenti finanziari.

## 🚀 Funzionalità

- **Analisi Tecnica Completa**: EMA 125, RSI 14, ATR, Historical Volatility, Multi-timeframe Momentum (ROC)
- **Analisi Sentiment**: Integrazione dati sentiment e news da EODHD API
- **Composite Scoring System**: Score combinato tecnico/sentiment con segnale operativo
- **Export JSON**: Dati strutturati per analisi con Agenti LLM (Claude, GPT, etc.)
- **Dashboard Interattiva**: Grafici Plotly, metriche dettagliate, news recenti

## 📦 Formati Ticker Supportati

| Tipo | Formato | Esempio |
|------|---------|---------|
| Equity US | SYMBOL.US | AAPL.US, MSFT.US |
| Equity Italia | SYMBOL.MI | ENI.MI, ISP.MI |
| Equity UK | SYMBOL.L | BP.L, HSBA.L |
| Equity Francia | SYMBOL.PA | TTE.PA, OR.PA |
| Equity Germania | SYMBOL.F | SAP.F, BMW.F |
| Crypto | SYMBOL-USD.CC | BTC-USD.CC, ETH-USD.CC |
| ETF | SYMBOL.US | SPY.US, QQQ.US |
| Index | SYMBOL.INDX | GSPC.INDX, DJI.INDX |

## 🛠️ Setup

### 1. Clona il Repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

### 2. Installa le Dipendenze

```bash
pip install -r requirements.txt
```

### 3. Configura la API Key EODHD

#### Per sviluppo locale

Crea un file `.streamlit/secrets.toml`:

```toml
EODHD_API_KEY = "your_api_key_here"
```

#### Per Streamlit Cloud

1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Seleziona la tua app
3. Vai su "Settings" → "Secrets"
4. Aggiungi:

```toml
EODHD_API_KEY = "your_api_key_here"
```

### 4. Avvia l'App

```bash
streamlit run app.py
```

## 📁 Struttura Repository

```
├── app.py                    # Applicazione Streamlit principale
├── requirements.txt          # Dipendenze Python
├── README.md                 # Questo file
└── .streamlit/
    └── config.toml           # Configurazione tema Streamlit
```

## 📊 Output JSON

L'app genera un file JSON strutturato per analisi con LLM contenente:

- **metadata**: Info asset, timestamp, quality check
- **current_price**: Prezzo e variazioni multi-periodo
- **technical_indicators**: EMA, RSI, Volume analysis
- **volatility_analysis**: ATR, HV, Regime detection
- **momentum_multitimeframe**: ROC 10/21/63, alignment
- **price_structure**: Range 52w, Drawdown, Livelli chiave
- **sentiment_analysis**: Score, dynamics, news velocity
- **composite_analysis**: Score finale, signal, confidence, key factors
- **llm_analysis_prompt**: Template pre-costruito per analisi LLM

## 🎯 Composite Score

Il sistema di scoring combina:

| Componente | Peso |
|------------|------|
| Technical Score | 55% |
| Sentiment Score | 45% |

Il Technical Score è composto da:
- EMA Component (35%)
- RSI Component (35%)
- Momentum Component (30%)

Adjustment Factors:
- Volume Confirmation (0.7 - 1.3)
- Volatility Adjustment (0.8 - 1.1)

## 📈 Segnali

| Score | Segnale | Descrizione |
|-------|---------|-------------|
| 80-100 | STRONG BUY | Convergenza positiva forte |
| 65-80 | BUY | Bias positivo |
| 45-65 | NEUTRAL | Segnali misti |
| 30-45 | SELL | Bias negativo |
| 0-30 | STRONG SELL | Convergenza negativa forte |

## ⚠️ Disclaimer

Questa analisi è fornita a scopo informativo e non costituisce consulenza finanziaria. Effettua sempre le tue ricerche prima di prendere decisioni di investimento.

## 📄 License

MIT License

## 🔗 Links

- **Kriterion Quant**: [kriterionquant.com](https://kriterionquant.com)
- **EODHD API**: [eodhd.com](https://eodhd.com)
- **Streamlit**: [streamlit.io](https://streamlit.io)

---

*Powered by Kriterion Quant | Data by EODHD*
