from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml.inference import engine
from api.routers import analise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega os pesos do modelo NLP na memória (1 vez no startup)
    engine.load_model()
    yield
    # Limpeza (se necessária) ao desligar

app = FastAPI(
    title="Fake News Hub API", 
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(analise.router)

@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "ml_ready": engine.hf_classifier is not None}

from fastapi.responses import HTMLResponse

@app.get("/mock-fake", tags=["teste"])
async def mock_fake():
    html = """
    <html>
      <head><title>URGENTE: Chá de alho cura o vírus</title></head>
      <body>
        <article>
          <h1>Urgente: Descoberta a cura!</h1>
          <p>ATENÇÃO! Repassem para todos os seus grupos! Um médico revelou que tomar água quente com limão mata o vírus em 24 horas. A mídia tradicional não quer que você saiba disso porque a indústria farmacêutica vai perder bilhões! O diretor do hospital já confirmou a informação em um áudio vazado no WhatsApp. Compartilhe antes que apaguem!</p>
        </article>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
