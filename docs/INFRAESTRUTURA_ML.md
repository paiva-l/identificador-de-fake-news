# Infraestrutura e Fluxo do Modelo de Machine Learning (ML) e Explicabilidade (XAI)

Este documento detalha como a nossa inteligência artificial funciona por debaixo dos panos, desde a recepção de uma URL até a extração da nota de confiabilidade e a geração matemática do comentário de explicação.

---

## 1. Visão Geral da Arquitetura (O Pipeline)

A nossa infraestrutura de ML foi concebida de forma assíncrona e em múltiplas camadas para não bloquear o serviço web (FastAPI) durante operações pesadas. O ciclo de vida de uma requisição segue o seguinte trajeto:

1. **Recepção e Cache:** A API recebe a URL. Antes de qualquer processamento, consulta o banco de dados (`SQLite`/`PostgreSQL`). Se a URL já foi analisada antes, devolve o resultado em milissegundos.
2. **Raspagem (Scraping):** Caso seja uma URL nova, uma rotina de extração assíncrona (usando `httpx` e `trafilatura`) faz o download do HTML. O *trafilatura* identifica inteligentemente onde está o corpo textual real da notícia, removendo banners, barras de navegação e menus.
3. **Higienização Inicial:** O texto extraído é limpo para remover URLs residuais, emojis e quebras de linha excessivas.
4. **Motor Híbrido de NLP:** O texto é injetado no `NLPEngine`, que processa a informação usando dois modelos distintos paralelamente:
   - **Módulo de Factualidade (LinearSVC):** Treinado para verificar indícios de desinformação.
   - **Módulo de Viés/Tom (mDeBERTa):** Avalia se a escrita é neutra/informativa ou sensacionalista/opinativa.
5. **Motor de Explicabilidade (XAI):** Recebe o escore de factualidade e os vetores de texto para traduzir a matemática do modelo em um texto legível para humanos.
6. **Armazenamento e Resposta:** O resultado completo (incluindo o texto da explicação) é gravado no banco de dados e enviado como JSON para a interface (Frontend).

---

## 2. Como os Modelos Avaliam o Texto?

### A. O Detector Factual (Fake News)
O coração da detecção de desinformação é um modelo linear de Máquinas de Vetores de Suporte (**LinearSVC**), calibrado e atrelado a um conversor **TF-IDF**. 
*   **Como funciona o TF-IDF?** Ele transforma o texto legível em uma "imagem matemática" (um vetor extenso). Ele conta quais palavras aparecem no texto, mas pune palavras muito comuns e dá maior pontuação a palavras raras e exclusivas.
*   **A Decisão Linear:** O `LinearSVC` mapeia essas palavras em um espaço matemático e traça uma linha reta separando os "pontos" de notícias verdadeiras dos "pontos" de notícias falsas.

### B. O Analisador de Viés (Zero-Shot)
Usamos um modelo massivo baseado na arquitetura Transformer, o **mDeBERTa-v3** (da Hugging Face). 
* Diferente do modelo anterior que busca padrões de vocabulário, o Transformer foca em "entender o contexto" semântico, comparando o texto com rótulos semânticos como: *"neutro, imparcial"* versus *"opinativo, sensacionalista"*.

---

## 3. A Mágica do Motor de Explicabilidade (XAI)

Como não utilizamos ferramentas externas geradoras de texto (LLMs como o ChatGPT) para inventar uma justificativa, nós **auditamos diretamente o cérebro matemático** do `LinearSVC`. 

O fluxo do XAI (`ExplainabilityEngine`) opera em 4 etapas estritamente determinísticas:

1. **Vetorização Exata:** Pegamos o texto limpo da notícia e o passamos pelo TF-IDF. O resultado é um vetor apontando quais palavras-chave existem naquela URL e a força (frequência) de cada uma.
2. **Resgate dos Pesos de Treinamento (`coef_`):** O `LinearSVC` guardou, ao longo de seu treinamento, o peso de cada uma das 50.000 palavras do vocabulário. Palavras associadas a Fake News receberam pesos severamente negativos (ex: "urgente", "espalhem", "cura", "farsa"). Palavras associadas a jornalismo real ganharam pesos positivos (ex: "segundo", "afirmou", "nesta terça-feira").
3. **Produto Escalar (Multiplicação):** A infraestrutura multiplica a força da palavra no texto (passo 1) pelo peso da palavra na memória do modelo (passo 2).
4. **Extração e Formatação:** 
   - Se o modelo diz que a notícia é falsa, o XAI vasculha o resultado da multiplicação procurando os maiores números negativos. Essas são as "Top Fake Words".
   - O XAI pega um molde de texto (*template* fixo baseado em regras) e o preenche com as palavras extraídas.
   - *Exemplo de Saída:* "Nossos algoritmos identificaram alta possibilidade de desinformação. Os termos que mais influenciaram o alerta foram: 'URGENTE', 'COMPARTILHEM', 'FRAUDE'."

---

## 4. Prontidão para Produção (Infraestrutura)

Respondendo à viabilidade comercial, **o backend (API) está completamente pronto** em sua essência. Para empacotar isso e apresentar ao usuário de ponta a ponta, recomenda-se a seguinte stack final de infraestrutura:

1. **Frontend (Onde o usuário cola a URL):** Uma interface em React.js, Vue.js, ou um simples aplicativo web/mobile que faz um POST via protocolo HTTP para o nosso endpoint `/analisar-url` mostrando carregadores visuais (loaders) enquanto os modelos calculam.
2. **Ambiente de Hospedagem da API:**
   - Como o modelo Hugging Face (`mDeBERTa`) carrega pesos complexos em memória (precisa de aproximadamente 1GB a 2GB de RAM para inicializar e realizar inferências velozes), hospedagens muito simples não servem.
   - O ideal é hospedar este projeto rodando em um conteiner **Docker** (já existe um `Dockerfile` e `docker-compose.yml` neste projeto) em uma máquina VPS como EC2 na AWS (t2.medium / t3.large), ou instâncias do Google Cloud Platform (GCP).
3. **Persistência de Cache:**
   - Atualmente, o SQLite cobre perfeitamente a prototipação. Em produção massiva, o SQLite é migrado facilmente para um banco relacional gerenciado como **PostgreSQL**, que já está previsto no `pyproject.toml` (dependências `asyncpg` e `psycopg2`).
