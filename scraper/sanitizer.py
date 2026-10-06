import re
import unicodedata

def sanitize_text(text: str) -> str:
    """
    Limpa o texto removendo links residuais, emojis e quebras de linha excessivas.
    """
    if not text:
        return ""

    # Remove URLs residuais
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove emojis (caracteres da categoria Unicode "Symbol, Other" - So)
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('So'))

    # Remove quebras de linha excessivas (3 ou mais viram apenas 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove espaços ou tabs excessivos na mesma linha
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()
