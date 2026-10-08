import re
import logging
import nltk

logger = logging.getLogger(__name__)

# Garantir que as stopwords estejam disponíveis no ambiente
try:
    from nltk.corpus import stopwords
    stopwords_pt = set(stopwords.words("portuguese"))
except LookupError:
    logger.info("Baixando NLTK stopwords...")
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stopwords_pt = set(stopwords.words("portuguese"))

def limpar_texto(texto: str) -> str:
    """
    Higieniza o texto bruto para que fique idêntico ao formato 
    que o modelo LinearSVC + TF-IDF encontrou durante o treinamento.
    """
    if not isinstance(texto, str):
        return ""

    # A. Converter para minúsculas
    texto = texto.lower()

    # B. Substituir URLs por um token reservado
    texto = re.sub(r"https?://\S+|www\.\S+", " urltoken ", texto)

    # C. Remover caracteres especiais e números (mantendo letras com acento)
    texto = re.sub(r"[^a-zA-Záàâãéèêíïóôõöúçñ\s]", " ", texto)

    # D. Tokenização por espaço e remoção de stopwords
    palavras = texto.split()
    palavras_limpas = [
        word for word in palavras if word not in stopwords_pt and len(word) > 2
    ]

    # E. Reagrupar texto
    return " ".join(palavras_limpas)
