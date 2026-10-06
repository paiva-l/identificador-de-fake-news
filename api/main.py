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
      <head><title>O papa Francisco foi preso após 80 acusações de tráfico</title></head>
      <body>
        <article>
          <h1>Boato – Ocorreu um apagão no Vaticano</h1>
          <p>O papa Francisco foi preso após 80 acusações de tráfico de crianças e diversas fraudes. Por isso que o pontífice está ausente. Apagão vaticano papar presar acusação tráfico criança e fraude. O papar francisco tuitou manhã o curador beaver confirmar o papar francisco equipar rede social agendar publicação e planejada antecedência clicar tweet o tweetdeck agendar o publicação. O fbi o interrogatório fontes oficiais militar policiar italiano e unidade crime sexual o casar papar vaticano prender e policiar alto e colocar prisão.</p>
        </article>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
