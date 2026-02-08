import json
from datetime import datetime

def prepare_json_download(analysis_data):
    """Formatta i dati dell'analisi in una stringa JSON."""
    if not analysis_data:
        return None
    return json.dumps(analysis_data, indent=2, ensure_ascii=False)

def generate_filename(ticker):
    """Genera un nome file basato su ticker e timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    ticker_clean = ticker.replace('.', '_').replace('-', '_')
    return f"{ticker_clean}_analysis_{timestamp}.json"
