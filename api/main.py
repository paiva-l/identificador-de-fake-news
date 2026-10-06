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
