import json
from datetime import datetime

def prepare_json_download(analysis_data):
    """
    Formatta i dati dell'analisi in una stringa JSON pronta per il download.
    """
    if not analysis_data:
        return None
    
    # Converte il dizionario in una stringa JSON formattata
    return json.dumps(analysis_data, indent=2, ensure_ascii=False)

def generate_filename(ticker):
    """
    Genera un nome file standardizzato basato sul ticker e sul timestamp.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    ticker_clean = ticker.replace('.', '_').replace('-', '_')
    return f"{ticker_clean}_analysis_{timestamp}.json"
