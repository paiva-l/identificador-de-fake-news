import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Lê a URL de ambiente (Postgres em prod/docker) ou recai para um SQLite local em dev.
# O PostgreSQL será: postgresql+asyncpg://user:pass@host:5432/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///./fake_news_local.db"
)

# Cria o engine assíncrono
engine = create_async_engine(DATABASE_URL, echo=False)

# Fábrica de sessões para injetar nas rotas FastAPI
AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    autoflush=False, 
    expire_on_commit=False
)

Base = declarative_base()
