# Kriterion Quant - Financial Sentiment Analysis Dashboard

Dashboard professionale per l'analisi quantitativa e di sentiment, sviluppata in Python e Streamlit.

## Deploy su Streamlit Cloud

Questa applicazione è configurata per leggere le API Key direttamente dai Secrets dell'ambiente Cloud, garantendo la massima sicurezza.

1.  Fai il Fork/Clone di questo repository su GitHub.
2.  Vai su [Streamlit Cloud](https://streamlit.io/cloud) e crea una nuova app collegata al tuo repository.
3.  Nelle "Advanced Settings" dell'app su Streamlit Cloud, inserisci il seguente secret nell'area **Secrets**:

```toml
EODHD_API_KEY = "LA_TUA_CHIAVE_API_EODHD"
