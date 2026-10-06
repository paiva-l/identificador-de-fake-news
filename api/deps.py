from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from .db.database import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency para prover a sessão do banco de dados assíncrono para as rotas.
    """
    async with AsyncSessionLocal() as session:
        yield session
