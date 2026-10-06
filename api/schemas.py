from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    url: HttpUrl

class AnalyzeResponse(BaseModel):
    url: str
    titulo: Optional[str]
    prob_fake: float
    taxa_confiabilidade: float
    bias_label: Optional[str]
    bias_score: Optional[float]
    modelos_usados: Optional[List[str]] = None
