# Model Card: Hub de Verificação de Notícias (Motor Híbrido NLP)

Este documento descreve a arquitetura, as interfaces e os limites operacionais do Motor Híbrido de Machine Learning desenvolvido para o MVP do Hub de Verificação de Notícias. Ele serve como contrato técnico para desenvolvedores Backend, Frontend e mantenedores da infraestrutura.

---

## 1. VISÃO GERAL (OVERVIEW)

### Propósito
O motor de Machine Learning atua como o principal módulo de validação de informações do sistema. Ele é acionado assim que uma URL de notícia é submetida e o texto é extraído pelo serviço de scraping. Seu objetivo é duplo:
1. **Detectar Padrões Falsos:** Avaliar a probabilidade matemática do conteúdo pertencer à classe de "Fake News".
2. **Análise de Viés e Nuances:** Identificar se a linguagem do artigo é factual e imparcial ou se possui forte carga opinativa e sensacionalista.

### Arquitetura Base do MVP (Motor Híbrido)
Para acelerar o time-to-market do MVP mantendo alta precisão em Português do Brasil, a solução implementa uma arquitetura **híbrida (Ensemble)** composta por dois modelos:

*   **Modelo Factual (LinearSVC Calibrado + TF-IDF):** 
    *   Treinado previamente em dados nativos (dataset *Fake.br / Recogna*) focados na gramática e nas gírias brasileiras de boatos. 
    *   *Justificativa:* Modelos estatísticos lineares (TF-IDF + SVC) são extremamente rápidos, requerem pouca RAM e captam com precisão os padrões semânticos de desinformação treinados localmente. Ele atua calibrado (`CalibratedClassifierCV`) para extração de probabilidade confiável.
*   **Modelo de Nuances (Zero-Shot Cross-lingual):**
    *   Utiliza o Transformer pré-treinado `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (via Hugging Face).
    *   *Justificativa:* Modelos Zero-Shot são excelentes generalistas que não exigem re-treinamento (fine-tuning) na fase de MVP. Como o mDeBERTa suporta múltiplos idiomas de forma robusta, ele é encarregado de classificar os sentimentos complexos (neutralidade vs. viés opinativo/sensacionalista) e atuar como _fallback_ caso o modelo linear falhe.

---

## 2. ESPECIFICAÇÕES DE ENTRADA E SAÍDA (API CONTRACT)

### Dado de Entrada Esperado
A inferência em si (módulo `ml.inference`) abstrai o link ou o HTML. O motor espera **apenas texto puro e sanitizado** (limpo de quebras de linha excessivas, links de navegação, ads e emojis).

**Limite Operacional:** Para evitar estouro de memória (OOM - Out of Memory) ou lentidão excessiva com a rede neural DeBERTa, o texto sofre truncamento (`text[:1500]`) ao ser processado no modelo de Viés. O LinearSVC, no entanto, processa o texto integralmente devido à sua velocidade.

### Payload de Resposta (JSON)
O modelo empacota os resultados devolvendo o seguinte esquema:

```json
{
  "url": "https://url-analisada.com.br",
  "titulo": "Título Extraído da Notícia",
  "prob_fake": 0.0018,
  "taxa_confiabilidade": 99.82,
  "bias_label": "neutro, imparcial, informativo",
  "bias_score": 0.5229,
  "modelos_usados": [
    "LinearSVC (BR)",
    "mDeBERTa-v3 (Zero-Shot)"
  ]
}
```

*   `prob_fake` *(Float 0.0 a 1.0)*: A probabilidade computada do texto ser mentiroso/boato.
*   `taxa_confiabilidade` *(Float)*: Um número derivado (ex: `1.0 - prob_fake`) amigável para exibição no Frontend.
*   `bias_label` *(String)*: A classificação categórica gerada pelo modelo Transformer. Geralmente os valores trafegam entre as chaves lógicas de *"neutro"* vs *"opinativo/enviesado"*.
*   `bias_score` *(Float 0.0 a 1.0)*: O grau de confiança da rede no `bias_label` apontado.

---

## 3. CICLO DE VIDA E INFRAESTRUTURA

### Carregamento Assíncrono (FastAPI Lifespan)
Redes Neurais (como o mDeBERTa) ocupam um bom volume de memória RAM (~1GB de pesos) e levam segundos para serem inicializadas. 
Para que as requisições web não sejam punidas e não haja bloqueio do evento loop do Uvicorn, o carregamento dos pesos na memória ocorre **uma única vez** no startup da aplicação, gerenciado pelo evento `lifespan` do FastAPI (`api.main`). 

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executado APENAS durante o boot da aplicação
    engine.load_model()
    yield
    # Cleanup (caso necessário)
```
Toda chamada ao `/analisar-url` utiliza o motor já "quente" na memória RAM (In-Memory Singleton), em thread assíncrona (`asyncio.to_thread`).

### Requisitos de Hardware Estimados
*   **CPU:** 2 vCPUs (A inferência em lote unitário sem GPU na nuvem tem processamento veloz com CPU moderna).
*   **RAM:** Mínimo de 1.5 GB disponíveis (Recomendado 2GB+ para o contêiner em nuvem) devido aos pesos do HuggingFace e concorrência web.
*   **Aceleração (GPU):** Não obrigatória para o MVP. O mDeBERTa processa payloads individuais rápidos o suficiente (< 2 segundos) em CPU.

### Gerenciamento de Ambiente e Docker
*   **Empacotamento:** Dependências são rigorosamente isoladas usando **Poetry**.
*   **Docker Multi-stage:** No build da imagem (`Dockerfile`), os pesos do modelo Hugging Face são interceptados e baixados preventivamente (*cache on build*). Isso garante que o conteiner final do Docker (`docker compose up`) suba instâncias escaláveis sem depender de comunicação externa à API do Hugging Face no momento do startup.

---

## 4. MÉTRICAS E THRESHOLDS (LIMITES DE DECISÃO)

A conversão do dado matemático (`prob_fake`) em categorias amigáveis de produto deve obedecer aos limiares de confiança da inteligência artificial:

*   **Verdadeira (Alta Confiança):** `prob_fake <= 0.35`
    *   *Ação:* Exibir selo verde de validado. A notícia tem padrões claríssimos de linguajar jornalístico/factual.
*   **Inconclusiva / Pendente de Revisão (Área Cinzenta):** `0.35 < prob_fake <= 0.65`
    *   *Ação:* Sinal de alerta (Amarelo). A linguagem mistura fatos com alto viés/sensacionalismo, ou o vocabulário não encontra base forte no treinamento. Aqui, o voto da comunidade passa a ser o fator decisivo para a validação.
*   **Falsa (Alta Probabilidade de Boato):** `prob_fake > 0.65`
    *   *Ação:* Sinalização vermelha. Estrutura vocabular fortemente alinhada a correntes falsas de WhatsApp e sites de clickbait malicioso.

*Recomendação para o Frontend:* Ao invés de exibir apenas o número exato, utilize _Banners_ de cores seguindo os thresholds acima.

---

## 5. LIMITAÇÕES E FEEDBACK LOOP (HUMAN-IN-THE-LOOP)

Nenhuma IA de NLP é imune a enganos. Para garantir a responsabilidade técnica e transparência do MVP, destacam-se as seguintes limitações conhecidas:

1.  **Limitação de Contexto Factual Recente:** O modelo LinearSVC e o mDeBERTa focam muito no **modo como a notícia foi escrita** (padrões gramaticais, estilo, carga de adjetivos) em vez de validar a verdade absoluta do evento do dia de hoje cruzando na base de dados do Google (Search/Fact-Checking). Se uma mentira for escrita de maneira brilhantemente parecida com uma matéria jornalística séria e formal, a IA pode "alucinar" aprovando-a.
2.  **Viés de Treinamento Original:** O LinearSVC atua bem com contexto sociopolítico coberto pelo dataset, mas neologismos repentinos de internet requerem renovação. O zero-shot lida muito bem com PT-BR, mas ainda possui um raciocínio _core_ majoritariamente anglo-saxônico.

### A Solução: Human-In-The-Loop (Votação Comunitária)
O pipeline foi projetado para contornar esses limites integrando as máquinas à comunidade. 

*   **Mecanismo de Resgate:** Quando uma URL tem sua predição cacheada no PostgreSQL (`noticias_cache`), abrimos espaço para a inserção na tabela `votos_comunidade`.
*   **Futuro Fine-Tuning:** Quando houver discrepância — a IA avalia como Verdade, mas a comunidade maciçamente relata e vota que é Falso — o sistema utilizará esse "desvio" acumulado no banco de dados para alimentar re-treinamentos mensais (Active Learning). As detecções humanas ajudarão a recalibrar os pesos das camadas do LinearSVC em iterações futuras (após o MVP de 7 dias), tornando o Hub progressivamente mais forte através de crowdsourcing.
