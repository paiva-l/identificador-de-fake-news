# 📰 Identificador de Fake News

Projeto de Processamento de Linguagem Natural (PLN) e Machine Learning desenvolvido para classificar notícias em **Verdadeiras** ou **Falsas** (dataset *Fake.br / Recogna*).

---

## 📁 Estrutura do Repositório

```text
identificador-de-fake-news/
├── data/
│   ├── fake_recogna.csv          # Dataset bruto
│   └── fake_recogna_limpo.csv    # Dataset sanitizado (Notebook 1)
├── models/
│   ├── modelo_regressao_logistica.pkl  # Modelo treinado (Notebook 2)
│   ├── vetorizador_tfidf.pkl           # Vetorizador TF-IDF
│   └── metrics_regressao_logistica.json # Métricas do modelo
├── 01_eda_limpeza_dados.ipynb     # Notebook 1: Análise e Limpeza
├── 02_treinamento_regressao_logistica.ipynb # Notebook 2: Treinamento
├── .gitignore
└── README.md

## 🛠️ Tecnologias Utilizadas
Python 3.10+

Pandas (Tratamento e manipulação de dados)

Scikit-Learn (TF-IDF, Regressão Logística e Métricas de Avaliação)

Seaborn / Matplotlib (Visualizações e Matriz de Confusão)

git clone [https://github.com/SEU_USUARIO/identificador-de-fake-news.git](https://github.com/paiva-l/identificador-de-fake-news.git)
cd identificador-de-fake-news