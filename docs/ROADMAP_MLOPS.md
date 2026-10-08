# Roadmap MLOps: Identificador de Fake News

Este documento traça o plano de evolução do projeto sob a perspectiva de **Machine Learning Operations (MLOps)**. O objetivo é transformar a arquitetura atual (que já é uma API funcional de Nível 1) em um sistema autônomo, rastreável e resiliente a mudanças no vocabulário de desinformação (Níveis 2 e 3).

---

## Estado Atual do Projeto
*   **Maturidade Atual:** Nível 1 (Serviço Integrado e Escalável).
*   **Conquistas:** API Assíncrona via FastAPI, extração de texto automatizada, modelo híbrido funcional, sistema de cache para reduzir latência e motor matemático de explicabilidade (XAI) isolado de IA generativa.
*   **Focos de Melhoria:** Desvio entre código de treino e inferência (*Training-Serving Skew*), falta de versionamento de dados/modelos e ausência de monitoramento sobre degradação do modelo ao longo do tempo.

---

## 🎯 Fase 1: Padronização e Estabilidade (Ações Imediatas)

O objetivo desta fase é eliminar as inconsistências entre o laboratório (Notebooks) e a produção (API) e preparar a infraestrutura para automação.

- [x] **1.1. Unificação do Pré-processamento NLP (Correção do *Training-Serving Skew*)**
  - **Problema:** A higienização de texto avançada (remoção de stopwords, regex) está isolada no notebook `01_preproc_eda.ipynb`. A API atualmente envia texto sujo para o Scikit-Learn.
  - **Ação:** Criar um módulo centralizado (ex: `ml/preprocessing.py`) contendo a função exata de `limpar_texto`.
  - **Integração:** Importar e utilizar essa função tanto antes de instanciar o `vectorizer.fit()` no treinamento, quanto antes do `predict()` no `ml/inference.py`.

- [x] **1.2. Otimização do Cold-Start do Hugging Face**
  - **Problema:** O `mDeBERTa-v3` baixa artefatos no *runtime*, retardando o tempo de resposta do primeiro usuário.
  - **Ação:** Criar um script de pré-download dos pesos e integrá-lo no processo de *build* do Docker. Os contêineres já devem nascer com os tensores na memória.

- [x] **1.3. Pipeline Base de Testes Contínuos (CI)**
  - **Ação:** Mapear os fluxos principais (extração, inferência e banco de dados) e adicionar GitHub Actions (ou similar) para rodar a suíte do `pytest` sempre que um novo código for empurrado (*push*) para a branch principal.

---

## 🎯 Fase 2: Governança de Modelos e Dados (Curto Prazo)

Nesta etapa, o projeto se desprende da dependência de grandes arquivos locais (`.joblib`, `.csv`) e adota ferramentas de rastreabilidade de Machine Learning.

- [ ] **2.1. Versionamento de Dados (DVC)**
  - **Ação:** Adicionar o **Data Version Control (DVC)** ao projeto. Arquivos como o `FakeRecogna.xlsx` e `fake_recogna_limpo.csv` deixarão de poluir o repositório Git, sendo versionados em buckets S3 ou Google Cloud Storage, enquanto o Git rastreia apenas os "ponteiros" `.dvc`.

- [ ] **2.2. Implementação do Model Registry (MLflow / W&B)**
  - **Ação:** Acoplar o **MLflow** ou o **Weights & Biases** aos scripts de treino (`notebooks/v2_pipeline_linearsvc/app.py`).
  - **Resultado:** Cada experimento de IA registrará métricas (F1-Score, Acurácia) em um servidor centralizado. A API da aplicação será alterada para *puxar* o último modelo marcado como "Production", eliminando a necessidade de substituição manual do `.joblib`.

---

## 🎯 Fase 3: Monitoramento Contínuo e Feedback (Longo Prazo)

Com a governança pronta, o projeto focará em monitorar o modelo operando no mundo real e criar gatilhos para que a IA aprenda continuamente.

- [ ] **3.1. Dashboards de Data Drift / Concept Drift**
  - **Problema:** A linguagem usada para criar desinformação evolui constantemente. Modelos estáticos envelhecem e perdem eficácia (*Data Drift*).
  - **Ação:** Usar ferramentas como **Evidently AI** para comparar o vetor semântico das notícias que estão chegando na API hoje com as notícias que treinaram o modelo meses atrás. Se houver desvio estatístico superior a 15%, emitir um alerta técnico no Slack/Email.

- [ ] **3.2. Volante de Dados (Feedback Loop) Automático**
  - **Ação:** Ativar as funcionalidades atreladas à tabela `VotoComunidade` existente no banco.
  - **Fluxo:** 
    1. A comunidade marca se a avaliação da máquina foi útil ou falhou.
    2. Registros conflitantes (Modelo diz Falso, Comunidade diz Verdadeiro) entram numa fila humana de revisão.
    3. As URLs validadas tornam-se novas linhas no dataset oficial.

- [ ] **3.3. Pipeline de Retreinamento Contínuo (CT - Continuous Training)**
  - **Ação:** Criar uma rotina (CRON) semanal ou mensal onde um contêiner baixa o dataset mais atual (passo 2.1), roda o script de MLflow (passo 2.2), limpa os dados (passo 1.1) e faz o re-treino total. Se as métricas do modelo novo superarem o modelo antigo, o deploy é automatizado em produção.
