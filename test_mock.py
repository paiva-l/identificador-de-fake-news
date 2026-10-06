import asyncio
import csv
from ml.inference import engine

async def main():
    fake_text = ""
    with open('fake_recogna_limpo.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Classe'] == '0':
                fake_text = row['texto_limpo']
                break
    print("TEXTO FAKE DO DATASET:", fake_text[:100])
    engine.load_model()
    try:
        res = engine.predict(fake_text)
        print("RESULTADO:", res)
    except Exception as e:
        print("ERRO:", e)

if __name__ == "__main__":
    asyncio.run(main())
