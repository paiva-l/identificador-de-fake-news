import httpx
import trafilatura
from .sanitizer import sanitize_text

class ExtractionError(Exception):
    """Exceção levantada quando a extração falha."""
    pass

def extract_news(url: str) -> dict:
    """
    Faz o download do HTML de uma URL e extrai título, autor e o corpo do texto limpo.
    """
    try:
        # Usa httpx com User-Agent comum para evitar bloqueios anti-bot básicos
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        with httpx.Client(follow_redirects=True, timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            downloaded = response.text
    except httpx.HTTPError as e:
        raise ExtractionError(f"Não foi possível acessar a URL. Erro: {e}")

    import json
    
    # Extrai dados da página ignorando lixo
    extracted_json = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        target_language="pt",
        output_format="json"
    )

    if not extracted_json:
        raise ExtractionError("Não foi possível extrair o texto principal da página.")

    try:
        result = json.loads(extracted_json)
    except json.JSONDecodeError:
        raise ExtractionError("Falha ao analisar o resultado da extração.")

    # Retorna o dicionário conforme o contrato
    return {
        "title": result.get('title') or '',
        "author": result.get('author') or '',
        "clean_content": sanitize_text(result.get('text', ''))
    }
