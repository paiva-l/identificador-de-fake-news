import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.deps import get_db
from api.db.models import NoticiaCache
from scraper.extractor import extract_news, ExtractionError
from ml.inference import engine

router = APIRouter(tags=["Análise"])

@router.post("/analisar-url", response_model=AnalyzeResponse)
async def analisar_url(request: AnalyzeRequest, session: AsyncSession = Depends(get_db)):
    url_str = str(request.url)
    
    # 1. Verificar cache no BD
    query = select(NoticiaCache).where(NoticiaCache.url == url_str)
    result = await session.execute(query)
    noticia_db = result.scalars().first()
    
    if noticia_db:
        prob = noticia_db.score_ml or 0.0
        return AnalyzeResponse(
            url=noticia_db.url,
            titulo=noticia_db.titulo,
            prob_fake=prob,
            taxa_confiabilidade=round((1.0 - prob) * 100, 2),
            bias_label=noticia_db.bias_label,
            bias_score=noticia_db.bias_score,
            modelos_usados=["Cache DB"]
        )

    # 2. Extração de texto usando Threads para não bloquear o loop assíncrono
    try:
        extracted_data = await asyncio.to_thread(extract_news, url_str)
    except ExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de extração: {e}")

    # 3. Inferência de ML em Thread para não bloquear
    try:
        prediction = await asyncio.to_thread(engine.predict, extracted_data["clean_content"])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência ML: {e}")

    # 4. Salvar no banco
    nova_noticia = NoticiaCache(
        url=url_str,
        titulo=extracted_data["title"],
        conteudo=extracted_data["clean_content"],
        score_ml=prediction["prob_fake"],
        bias_label=prediction["bias_label"],
        bias_score=prediction["bias_score"]
    )
    
    session.add(nova_noticia)
    await session.commit()
    await session.refresh(nova_noticia)

    # 5. Retornar resposta
    prob = prediction["prob_fake"]
    return AnalyzeResponse(
        url=nova_noticia.url,
        titulo=nova_noticia.titulo,
        prob_fake=prob,
        taxa_confiabilidade=round((1.0 - prob) * 100, 2),
        bias_label=prediction["bias_label"],
        bias_score=prediction["bias_score"],
        modelos_usados=prediction["modelos_usados"]
    )
