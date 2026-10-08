# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependências de sistema necessárias para compilação (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copiar arquivos de dependência
COPY pyproject.toml poetry.lock ./

# Exportar dependências para requirements.txt para instalação limpa
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Instalar dependências no ambiente global do builder
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Imagem Final
FROM python:3.11-slim

WORKDIR /app

# Instalar bash (útil para scripts de entrypoint) e dependências de runtime do postgres (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências instaladas no builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# Criar diretório para cache do HuggingFace
RUN mkdir -p /app/.cache/huggingface

# Pré-baixar os pesos do HuggingFace (mDeBERTa) e NLTK stopwords para evitar Cold Start
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')"
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copiar o código da aplicação
COPY . .

# Expor a porta do FastAPI
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
