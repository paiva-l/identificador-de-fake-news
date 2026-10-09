# Identificador de Fake News

Projeto de PLN da Residência em Inteligência Artificial — UnB: baseline TF-IDF, Stacking, validação externa, ablação com retreinamento e inferência por URL. Classificação textual FAKE/REAL não substitui verificação factual independente.

## Instalação

```powershell
git clone https://github.com/paiva-l/identificador-de-fake-news.git
cd identificador-de-fake-news
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name identificador-fake-news
.venv\Scripts\python -m jupyter lab
```

Selecione esse kernel e execute as células na ordem. Recursos NLTK e extração de notícias exigem internet. As dependências permitem retreinamento; modelos serializados históricos podem exigir as versões registradas no respectivo bundle.

## Dados

A entrada original já versionada é `data/FakeRecogna.xlsx`; `data/fake_recogna_limpo.csv` é a versão limpa. Treinamento: **0 = FAKE, 1 = REAL**.

O externo `data/fakerecogna_extrativo.csv` permanece local. Para reproduzir validação externa e triagem V3, obtenha a mesma cópia usada no experimento e coloque-a nesse caminho. Não há endereço público de download confirmado neste repositório. Colunas obrigatórias: `Titulo`, `Noticia`, `Autor`, `Label`. Rótulos externos: **1 = FAKE, 0 = REAL**, convertidos pelo notebook 03. Uma base diferente requer nova auditoria.

## Ordem de execução

1. V1: `notebooks/v1_baseline/01_limpeza_eda_pln_fakerecogna.ipynb`, `02_avaliacao_modelo_regressao_logistica.ipynb` e `03-benchmark-de-modelos.ipynb`. O modo revisado exige revisão manual em `data/audit/`; consulte as opções iniciais. O benchmark exploratório utiliza a base original.
2. V2: em `notebooks/v2_validacao_externa/`, execute `01_preproc_eda.ipynb`, `02_treinamento_ensemble_stacking.ipynb`, `03_validacao_externa_ood.ipynb`, `04_diagnostico_valicao_externa.ipynb` e `05_ablacao.ipynb`, nessa ordem. O diagnóstico pode consultar modelos do benchmark V1. A ablação depende dos registros produzidos por 02 e 03.
3. Auditoria opcional V1: `04_auditoria_dos_textos_de_checagem.ipynb`; treine V2 antes das etapas que consultam Stacking.
4. V3: `notebooks/v3_teste_inferencia_dados_nao_vistos.ipynb`, após gerar a variante `sem_autoria` no 05 e os registros anteriores. A triagem também requer o dataset externo.

Mantenha os módulos `.py` junto aos notebooks. Modelos são gerados em `models/`; partições e manifestos na pasta de dados V2; relatórios de execução em `reports/`. Esses arquivos são regenerados e não entram no novo envio. Para inferência sem retreinamento, recupere os bundles e registros compatíveis de uma cópia local.

Os notebooks publicados preservam as saídas disponíveis: gráficos, tabelas de métricas, matrizes de confusão e textos Markdown. Para recalcular os resultados, execute as células na ordem indicada.

## Documentação

- [Relatório técnico](reports/relatorio_tecnico.md)
- [Síntese V1–V3](reports/relatorio_tecnico_modelo_v1_v2_v3.md)

Holdout e externo já observados não constituem novo teste independente. V4 supervisionada, DVC, MLflow e implantação em nuvem são propostas.

Caches, modelos, dados intermediários, revisão manual e resultados extensos permanecem locais. Artefatos presentes em commits anteriores continuam no histórico Git.