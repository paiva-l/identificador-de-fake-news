import asyncio
from scraper.extractor import extract_news
from ml.inference import engine

async def main():
    url = "https://sensacionalista.com.br/2023/10/06/homem-e-preso-por-nao-ter-opiniao-sobre-assunto-polemico/"
    engine.load_model()
    try:
        data = extract_news(url)
        print("EXTRAÍDO:", data["clean_content"][:200])
        res = engine.predict(data["clean_content"])
        print("RESULTADO:", res)
    except Exception as e:
        print("ERRO:", e)

if __name__ == "__main__":
    asyncio.run(main())
