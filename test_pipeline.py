import sys
from scraper.extractor import extract_news
from ml.inference import engine

def main(url):
    print(f"Extraindo notícia da URL: {url}")
    try:
        data = extract_news(url)
    except Exception as e:
        print(f"Erro na extração: {e}")
        sys.exit(1)
        
    print(f"\nTítulo: {data['title']}")
    print(f"Conteúdo limpo (amostra): {data['clean_content'][:200]}...")
    
    print("\nCarregando modelos (isso pode demorar na 1ª vez)...")
    engine.load_model()
    
    print("\nExecutando inferência...")
    result = engine.predict(data['clean_content'])
    
    print("\n=== RESULTADO ===")
    print(f"Probabilidade de ser Fake: {result['prob_fake']*100:.2f}%")
    print(f"Taxa de Confiabilidade: {(1.0 - result['prob_fake'])*100:.2f}%")
    print(f"Viés/Nuance: {result['bias_label']} (Score: {result['bias_score']:.4f})")
    print(f"Modelos Usados: {', '.join(result['modelos_usados'])}")

if __name__ == "__main__":
    test_url = "https://g1.globo.com/politica/noticia/2024/02/08/operacao-da-pf-mira-bolsonaro-braga-netto-e-valdemar-costa-neto.ghtml"
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    main(test_url)
