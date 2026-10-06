import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship

from .database import Base

class NoticiaCache(Base):
    __tablename__ = 'noticias_cache'

    id = Column(Integer, primary_key=True, index=True)
    # A URL será indexada para consultas rápidas na Etapa 4
    url = Column(String, unique=True, index=True, nullable=False)
    titulo = Column(String, nullable=True)
    conteudo = Column(Text, nullable=True)
    # A probabilidade de ser Fake News calculada pelo motor de ML
    score_ml = Column(Float, nullable=True)
    # Resultados do modelo de viés (Hugging Face)
    bias_label = Column(String, nullable=True)
    bias_score = Column(Float, nullable=True)
    # Registro de quando a URL foi extraída e classificada
    data_processamento = Column(DateTime, default=datetime.datetime.utcnow)

    # Relacionamento 1-N com os votos da comunidade
    votos = relationship("VotoComunidade", back_populates="noticia")


class VotoComunidade(Base):
    __tablename__ = 'votos_comunidade'

    id = Column(Integer, primary_key=True, index=True)
    id_noticia = Column(Integer, ForeignKey('noticias_cache.id'), nullable=False)
    # Identificador anonimizado do usuário para evitar fraudes/brigading
    id_usuario = Column(String, index=True, nullable=False)
    # O voto da pessoa (-1, 0, 1 etc.)
    voto = Column(Integer, nullable=False)
    # Para futuros refinamentos de reputação
    peso_voto = Column(Float, default=1.0)

    noticia = relationship("NoticiaCache", back_populates="votos")
