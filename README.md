# 📰 Identificador de Fake News (Motor NLP Híbrido & MLOps)

Bem-vindo ao repositório do **Identificador de Fake News**, um projeto avançado de Processamento de Linguagem Natural (NLP) focado na detecção de desinformação em textos jornalísticos e artigos da web. 

Este projeto evoluiu de uma simples experimentação de laboratório para uma **Arquitetura de API de Produção Escalável** com foco completo em **MLOps**, Automação e Explicabilidade (XAI).

---

## 🚀 Principais Funcionalidades

1. **Inteligência Híbrida (Ensemble):**
   * **Motor Factual (LinearSVC):** Treinado e calibrado em bases de dados brasileiras (*FakeRecogna*), analisa a distribuição matemática TF-IDF do texto para detectar padrões de mentira ou boato.
   * **Motor de Viés (mDeBERTa-v3):** Um modelo de Hugging Face *Zero-Shot* que avalia se a escrita apela para sensacionalismo ou se mantém uma linguagem jornalística neutra.

2. **Explicabilidade Matemática (XAI Livre de Alucinação):**
   * Sem o uso de IA Generativa (ChatGPT/LLMs), o motor *ExplainabilityEngine* intercepta os coeficientes vetoriais que a máquina gerou. Ele aponta com exatidão matemática quais as palavras que tornaram a notícia *Fake* ou *Verdadeira*, gerando a janela de análise detalhada exigida pelo Front-end.

3. **Governança e MLOps 100% Integrados:**
   * **DVC & MLflow:** Versionamento inteligente de bases de treino e registros de modelos matemáticos.
   * **Continuous Training (CRON):** Rotina `retrain_cron.py` que ouve o *feedback* da comunidade (via BD) e retreina a Inteligência Artificial automaticamente.
   * **Data Drift (`drift_monitor.py`):** Monitora em tempo real se a estrutura linguística das Fake News modernas está escapando do padrão aprendido no laboratório original, emitindo relatórios via *Evidently AI*.

4. **Extração Web Autônoma e Cache:**
   * Basta enviar uma URL. O sistema extrai o texto principal descartando menus e banners (*newspaper3k*), e grava inferências idênticas em um cache no PostgreSQL/SQLite.

---

## 🏗️ Estrutura do Projeto

```text
identificador-de-fake-news/
├── api/                   # Aplicação web (FastAPI, rotas, banco de dados, schemas)
├── ml/                    # Lógica do Machine Learning (Treino, Inferência, XAI, MLOps)
├── scraper/               # Motor de web scraping de notícias
├── models/                # Artefatos serilizados do Scikit-Learn (.joblib)
├── docs/                  # ROADMAP MLOps e guias de arquitetura
├── docker-compose.yml     # Orquestrador de contêineres de produção
└── Dockerfile             # Imagem encapsulada com pré-download de Hugging Face
```

---

## 💻 Como Rodar a API (via Docker)

Não é necessário instalar bibliotecas pesadas de Python localmente. Tudo já foi conteinerizado para o máximo desempenho de isolamento.

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/identificador-de-fake-news.git
   cd identificador-de-fake-news
   ```

2. **Inicie o servidor via Docker Compose:**
   ```bash
   docker-compose up -d --build
   ```
   > *Nota: O primeiro `build` fará o pré-download dos modelos do Hugging Face para dentro do contêiner, o que pode levar alguns minutos (otimização de Cold-Start).*

3. **Acesse a Documentação (Swagger):**
   Acesse no seu navegador: `http://localhost:8000/docs`

---

## 📡 Utilização da API (Exemplo)

Envie uma requisição HTTP **POST** para `/analisar-url`:

```json
{
  "url": "https://g1.globo.com/exemplo-de-noticia.ghtml"
}
```

O servidor extrai, classifica e exibe toda a auditoria transparente na resposta:

```json
{
  "url": "https://g1.globo.com/exemplo-de-noticia.ghtml",
  "prob_fake": 0.12,
  "taxa_confiabilidade": 88.0,
  "janela_analise": {
    "evidencia": "A análise lexical sugere forte alinhamento com reportagens factuais.",
    "qualidade_da_fonte": "O domínio g1.globo.com é reconhecido como veículo oficial.",
    "corroboracao": "A linguagem parece neutra.",
    "contexto": "A notícia analisada possui 450 palavras processadas.",
    "atualidade": "Requer validação temporal estática externa."
  },
  "o_que_sustenta": [
    "Linguagem consistente: repórter",
    "Linguagem consistente: confirmou"
  ],
  "modelos_usados": ["LinearSVC (BR)", "mDeBERTa-v3 (Zero-Shot)"]
}
```

---

## 🤝 Voto Comunitário (Melhoria Contínua)
A API suporta receber auditoria humana via `POST /votar`, onde votos conflitantes geram retreino semanal automático das predições do modelo.

> O roadmap completo das fases de MLOps encontra-se na pasta `docs/ROADMAP_MLOPS.md`.
