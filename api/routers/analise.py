import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.schemas import AnalyzeRequest, AnalyzeResponse, JanelaAnalise
from api.deps import get_db
from api.db.models import NoticiaCache
from scraper.extractor import extract_news, ExtractionError
from ml.inference import engine

router = APIRouter(tags=["Análise"])

def dict_to_response(url: str, titulo: str, prob_fake: float, bias_label: str, bias_score: float, xai_dict: dict, modelos_usados: list):
    janela = xai_dict.get("janela_analise", {})
    return AnalyzeResponse(
        url=url,
        titulo=titulo,
        prob_fake=prob_fake,
        taxa_confiabilidade=round((1.0 - prob_fake) * 100, 2),
        bias_label=bias_label,
        bias_score=bias_score,
        janela_analise=JanelaAnalise(
            evidencia=janela.get("evidencia", ""),
            qualidade_da_fonte=janela.get("qualidade_da_fonte", ""),
            corroboracao=janela.get("corroboracao", ""),
            contexto=janela.get("contexto", ""),
            atualidade=janela.get("atualidade", "")
        ),
        o_que_sustenta=xai_dict.get("o_que_sustenta", []),
        o_que_enfraquece=xai_dict.get("o_que_enfraquece", []),
        o_que_nao_foi_confirmado=xai_dict.get("o_que_nao_foi_confirmado", []),
        modelos_usados=modelos_usados
    )

@router.post("/analisar-url", response_model=AnalyzeResponse)
async def analisar_url(request: AnalyzeRequest, session: AsyncSession = Depends(get_db)):
    url_str = str(request.url)
    
    # 1. Verificar cache no BD
    query = select(NoticiaCache).where(NoticiaCache.url == url_str)
    result = await session.execute(query)
    noticia_db = result.scalars().first()
    
    if noticia_db:
        prob = noticia_db.score_ml or 0.0
        try:
            xai_dict = json.loads(noticia_db.explicacao_xai) if noticia_db.explicacao_xai else {}
        except:
            xai_dict = {}
        return dict_to_response(noticia_db.url, noticia_db.titulo, prob, noticia_db.bias_label, noticia_db.bias_score, xai_dict, ["Cache DB"])

    # 2. Extração de texto usando Threads para não bloquear o loop assíncrono
    try:
        extracted_data = await asyncio.to_thread(extract_news, url_str)
    except ExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de extração: {e}")

    # 3. Inferência de ML em Thread para não bloquear
    try:
        prediction = await asyncio.to_thread(engine.predict, extracted_data["clean_content"], url_str)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência ML: {e}")

    xai_dict = prediction.get("explicacao_xai") or {}
    
    # 4. Salvar no banco
    nova_noticia = NoticiaCache(
        url=url_str,
        titulo=extracted_data["title"],
        conteudo=extracted_data["clean_content"],
        score_ml=prediction["prob_fake"],
        bias_label=prediction["bias_label"],
        bias_score=prediction["bias_score"],
        explicacao_xai=json.dumps(xai_dict, ensure_ascii=False)
    )
    
    session.add(nova_noticia)
    await session.commit()
    await session.refresh(nova_noticia)

    # 5. Retornar resposta
    prob = prediction["prob_fake"]
    return dict_to_response(
        url=nova_noticia.url,
        titulo=nova_noticia.titulo,
        prob_fake=prob,
        bias_label=prediction["bias_label"],
        bias_score=prediction["bias_score"],
        xai_dict=xai_dict,
        modelos_usados=prediction["modelos_usados"]
    )

from api.schemas import FeedbackRequest
from api.db.models import VotoComunidade

@router.post("/votar")
async def votar(request: FeedbackRequest, session: AsyncSession = Depends(get_db)):
    url_str = str(request.url)
    
    # Busca a notícia no cache para vincular o voto
    query = select(NoticiaCache).where(NoticiaCache.url == url_str)
    result = await session.execute(query)
    noticia_db = result.scalars().first()
    
    if not noticia_db:
        raise HTTPException(status_code=404, detail="Notícia não encontrada no sistema para votar.")
        
    # Registrar o voto
    novo_voto = VotoComunidade(
        id_noticia=noticia_db.id,
        id_usuario=request.id_usuario,
        voto=request.voto
    )
    
    session.add(novo_voto)
    await session.commit()
    
    return {"status": "success", "message": "Voto computado com sucesso. Obrigado por ajudar a melhorar nosso modelo!"}
