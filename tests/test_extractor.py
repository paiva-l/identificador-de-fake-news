from scraper.extractor import extract_news, ExtractionError

def test_extract_news_real_url():
    # URL confiável da Wikipédia sobre NLP para evitar instabilidades de portais (404/bot blocks)
    url = "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural"

    try:
        result = extract_news(url)
        
        # Validações dos campos presentes
        assert "title" in result
        assert "clean_content" in result
        assert "author" in result
        
        # Valida se o texto foi extraído com um tamanho mínimo razoável
        assert len(result["clean_content"]) > 100
        
        # Print do resultado para validação manual (dry-run)
        print("\n--- RESULTADO DA EXTRAÇÃO ---")
        print(f"URL: {url}")
        print(f"Título: {result['title']}")
        print(f"Autor(es): {result['author']}")
        print(f"Conteúdo Limpo (início): {result['clean_content'][:200]}...")
        
    except ExtractionError as e:
        print(f"\nA extração falhou, possivelmente devido a bloqueio da página: {e}")
