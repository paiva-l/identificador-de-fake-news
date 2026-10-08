from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any

class AnalyzeRequest(BaseModel):
    url: HttpUrl

class JanelaAnalise(BaseModel):
    evidencia: str
    qualidade_da_fonte: str
    corroboracao: str
    contexto: str
    atualidade: str

class AnalyzeResponse(BaseModel):
    url: str
    titulo: Optional[str]
    prob_fake: float
    taxa_confiabilidade: float
    bias_label: Optional[str]
    bias_score: Optional[float]
    janela_analise: JanelaAnalise
    o_que_sustenta: List[str]
    o_que_enfraquece: List[str]
    o_que_nao_foi_confirmado: List[str]
    modelos_usados: Optional[List[str]] = None

class FeedbackRequest(BaseModel):
    url: HttpUrl
    id_usuario: str
    voto: int  # 1 (Verdadeira), 0 (Fake), -1 (Discorda do Modelo)
