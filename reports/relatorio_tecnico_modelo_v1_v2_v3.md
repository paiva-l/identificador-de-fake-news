# Relatório Técnico — Pipeline de Classificação de Fake News

**Programa:** Residência em Inteligência Artificial — Universidade de Brasília (UnB)  
**Projeto:** Identificador de Fake News / Hub de Verificação de Notícias  
**Data de referência:** 9 de outubro de 2026  
**Escopo:** preparação de dados, modelagem, avaliação, explicabilidade e operação de modelos de processamento de linguagem natural (NLP).

## Resumo do projeto

O projeto Identificador de Fake News / Hub de Verificação de Notícias investiga a classificação de notícias em português para apoiar a triagem de conteúdos potencialmente desinformativos e a revisão humana. O desenvolvimento documentado abrange preparação e auditoria de dados, modelos baseline com TF-IDF, um Stacking Ensemble com Regressão Logística, Random Forest e LinearSVC, avaliação interna e externa, diagnóstico de erros, ablação com retreinamento e inferência por URL com extração via Trafilatura. A rastreabilidade é apoiada por manifestos, hashes, modelos salvos e registros dos experimentos.

Este relatório distingue implementação observável, evolução proposta e resultados hipotéticos. Os resultados quantitativos apresentados para V1, V2 e V3 foram extraídos dos registros locais e identificados por etapa e execução. Versionamento com DVC, API FastAPI e análise de tom com mDeBERTa em modo zero-shot são mencionados no documento de referência fornecido, mas sua implementação não foi confirmada neste workspace. Fine-tuning supervisionado de transformers para classificação de Fake News, integração do treinamento com MLflow e implantação em AWS/GCP são possibilidades de evolução arquitetural, não resultados demonstrados pelos experimentos aqui descritos. Eventuais comparações hipotéticas entre famílias de modelos devem ser identificadas como simulações e separadas das métricas efetivamente medidas.

A principal restrição científica é que padrões linguísticos associados ao rótulo FAKE não comprovam a falsidade de uma alegação. A solução deve comunicar essa diferença e complementar a classificação textual com evidências externas e revisão especializada. O funcionamento técnico do pipeline e a intensidade de seus escores não substituem a verificação factual independente.

## 1. Síntese dos resultados

O sistema classifica padrões textuais em duas classes: **0 = Fake e 1 = Real**. Não realiza, por si só, verificação de fatos. O desenvolvimento passou de modelos baseline da V1 para um Stacking Ensemble da V2, inicialmente com título, notícia e autor. Após resultados internos elevados e perda de desempenho na avaliação externa, foram investigadas dependência de autoria, diferenças entre corpora, erros por classe e sensibilidade das entradas.

O stacking completo apresentou F1-macro OOF de **0,9900**, F1-macro no holdout de **0,9933** e F1-macro externo de **0,7675**. Na execução completa anterior da ablação, a versão retreinada com **Título + Notícia, sem autoria**, alcançou o maior F1-macro externo entre as quatro variantes: **0,8437**. A Random Forest individual da V2 obteve **0,8651** no mesmo corpus externo, portanto o stacking completo não superou todas as suas bases nesse conjunto.

A V3 carregou a variante sem autoria e classificou uma reportagem da VEJA como **REAL**, com score de **8,602148**. O registro confirma a conclusão das verificações técnicas daquela execução. Não há veredito factual documentado que permita afirmar se a classificação corresponde à realidade.

## 2. Evidências e limites deste relatório

Os resultados foram reunidos a partir dos notebooks, utilitários, manifestos e relatórios locais; não houve novo treinamento para produzir este documento. Os caminhos das fontes aparecem na seção 14.

Durante a leitura, `ablacao/comparacao_metricas.csv` continha somente duas linhas OOF, compatíveis com a regravação incremental de uma nova execução. A comparação completa apresentada aqui foi recuperada de `ablacao/interpretacao.md` e `ablacao/conclusao_ablacao.md`, que preservam uma execução anterior. **Esses números não são apresentados como conclusão da reexecução parcial atual.** É necessário conferir os hashes e concluir a nova execução antes de combinar seus artefatos com avaliações anteriores.

O episódio do FACTCKBR é descrito conforme esclarecimento fornecido pela responsável pelo projeto. Não foram localizados os arquivos de predição, métricas ou logs dessa tentativa no estado atual do workspace e no histórico versionado examinado. Por isso, o relatório documenta a incompatibilidade relatada, mas não atribui resultados numéricos ao FACTCKBR.

## 3. Dados, rótulos e organização

### 3.1 Corpus de desenvolvimento

O dataset de referência é `data/FakeRecogna.xlsx`. Os campos utilizados pelos pipelines são `Titulo`, `Noticia`, `Autor` e o rótulo original `Classe`. Após preparação, os registros incluem `id_original`, texto composto e `label`.

O manifesto atual da V2 registra **9.518 exemplos de treino e 2.384 exemplos de holdout**, totalizando **11.902**. O holdout-alvo é de 20%, com semente 42. O manifesto registra separação em modo similaridade, limiar 0,9, seis vizinhos e 262.144 atributos de hashing. Há validações para texto vazio, rótulo inválido, IDs duplicados, duplicatas normalizadas e consistência entre o texto composto e as colunas de origem.

### 3.2 Corpus externo extrativo

O arquivo `data/fakerecogna_extrativo.csv` possui **52.800 registros originais**. A auditoria reteve **46.229** para avaliação, excluindo **6.571**. São verificadas coincidências de URL, título, texto e similaridade elevada em relação ao corpus original, além de condições de qualidade das entradas.

O extrativo é um corpus derivado do FakeRecogna. A filtragem automática não certifica independência por evento, fonte ou origem. Seus resultados são descritos como **robustez em corpus derivado**, e não como certificação de generalização OOD independente.

### 3.3 Conversão dos rótulos

| Classe | Desenvolvimento e modelo | `Label` no CSV externo original |
|---|---:|---:|
| Fake | 0 | 1 |
| Real | 1 | 0 |

Antes de avaliar o corpus externo, aplica-se o mapeamento `{1: 0, 0: 1}`. O holdout já usa a codificação do treinamento e não deve receber essa inversão. O notebook 05 compara os rótulos convertidos com as previsões e referências persistidas pelo notebook 03.

### 3.4 Estrutura de arquivos

Os datasets de entrada foram centralizados em `data/`. Versões diferentes do dataset limpo foram preservadas com nomes distintos. Os modelos algorítmicos foram centralizados em `models/`, com subpastas para os experimentos atuais e legados. Os vetorizadores que haviam sido movidos foram devolvidos aos locais anteriores, conforme o escopo solicitado.

A pasta V2 foi renomeada para `notebooks/v2_validacao_externa`, e as referências encontradas no código e na configuração do projeto foram atualizadas. As partições processadas continuam em `notebooks/v2_validacao_externa/data/processed/texto_autor_exploratorio_v4/`.

O `manifest.json` associa esquema, política de entrada, rótulos, configuração da separação e hashes SHA-256 dos dados. Esses controles interrompem avaliações quando os artefatos não correspondem à mesma execução.

## 4. V1 — Limpeza, baseline e auditoria

A V1 contém limpeza e análise exploratória, avaliação de Regressão Logística, benchmark de algoritmos e auditoria dos textos de checagem. As funções compartilhadas estão em `baseline_features.py`; os gráficos são apoiados por `evaluation_plots.py`.

| Modelo no benchmark V1 | Acurácia | F1-macro |
|---|---:|---:|
| LinearSVC | 0,9861 | 0,9861 |
| Regressão Logística | 0,9794 | 0,9794 |
| Random Forest | 0,9769 | 0,9769 |
| MultinomialNB | 0,9735 | 0,9735 |

O LinearSVC teve o maior F1-macro nesse benchmark. A comparação V1/V2 não isola apenas a troca de algoritmo: protocolos e partições não devem ser tratados como equivalentes sem conferência.

O modo exploratório carrega a base original sem exigir aprovação manual integral. No modo estrito, a ficha `data/audit/v1_revisao_texto_autor.csv` é obrigatória; se ausente, o utilitário cria um modelo de ficha e interrompe o treinamento até a revisão.

O resumo de triagem de 8 de outubro de 2026 registra 11.902 exemplos e **3.648 com marcador editorial explícito**. Houve consulta inicial de dez registros selecionados por ordem: sete fontes acessíveis e três falhas de acesso. Essa seleção não é amostra aleatória nem revisão integral. Termos de checagem e autoria do checador podem revelar o processo editorial sem representar a alegação original. A falta de termo sinalizador não equivale a aprovação do exemplo.

## 5. V2 — Arquitetura e protocolo de treinamento

### 5.1 Preparação das features

Título e corpo são combinados em uma entrada textual. A normalização inclui tratamento Unicode, entidades HTML, padronização para minúsculas, remoção de marcação HTML, substituição de URLs e normalização de caracteres e espaços.

Cada base utiliza seu próprio TF-IDF, limitado a 20.000 atributos e n-gramas de uma e duas palavras. O autor é normalizado e codificado com one-hot; autores desconhecidos são ignorados na transformação. Valores ausentes ou semelhantes a datas recebem a categoria de autor desconhecido.

### 5.2 Classificadores

| Componente | Configuração principal |
|---|---|
| Base Regressão Logística | `max_iter=2000`, semente 42 |
| Base Random Forest | 200 árvores, semente 42, `n_jobs=-1` |
| Base LinearSVC | `dual=True`, `max_iter=10000`, semente 42 |
| Meta-modelo | Regressão Logística, `max_iter=2000`, semente 42 |
| Stacking | `stack_method="auto"`, `passthrough=False`, `n_jobs=1` |

O meta-modelo recebe saídas das bases, não diretamente as features originais. O método automático usa a saída disponível em cada base, como probabilidades ou margens. As previsões internas out-of-fold treinam o meta-modelo.

### 5.3 Validação aninhada e prevenção de contaminação direta

O desenvolvimento utiliza cinco dobras externas estratificadas, com embaralhamento e semente 42. Dentro de cada stacking há outras cinco dobras internas com a mesma configuração. Cada pipeline base contém seu próprio processamento; o vocabulário, IDF e categorias de autoria são ajustados dentro das dobras pertinentes.

Cada registro recebe exatamente uma previsão OOF externa. O ajuste final ocorre no treino completo. O treinamento do notebook 02 não utiliza o holdout ou o conjunto externo para ajuste. Avisos de convergência são tratados como erro para evitar concluir silenciosamente ajustes não convergidos.

### 5.4 Resultados OOF V2

| Modelo | F1-macro OOF | Fake como Real | Real como Fake |
|---|---:|---:|---:|
| Stacking | 0,9900 | 57 | 38 |
| LinearSVC | 0,9878 | 83 | 33 |
| Regressão Logística | 0,9770 | 181 | 38 |
| Random Forest | 0,9742 | 142 | 104 |

O stacking teve o melhor F1-macro interno. O desvio do F1-macro entre dobras foi de aproximadamente 0,00255; ele não representa intervalo de confiança. O ajuste final registrado durou 68,41 segundos, e os ajustes das cinco dobras externas totalizaram 262,15 segundos, na máquina e execução documentadas.

## 6. FACTCKBR — Incompatibilidade do domínio de entrada

**Segundo o relato da responsável, o problema observado com FACTCKBR não foi um erro de código.** O stacking foi treinado para artigos jornalísticos completos, utilizando título, corpo da notícia e autoria. Já os exemplos usados na tentativa com FACTCKBR continham principalmente alegações curtas verificadas por agências de fact-checking.

Essa diferença altera a unidade textual, o comprimento, o contexto, a distribuição lexical e a disponibilidade de features. Trata-se de uma incompatibilidade entre o domínio de entrada esperado pelo modelo e o domínio apresentado na avaliação, compatível com **mudança de distribuição — distribution shift**. Converter nomes de colunas ou repetir uma alegação curta em campos diferentes não transforma esse material em uma notícia completa.

Um pipeline pode executar sem exceções e ainda produzir resultados pouco confiáveis fora de seu domínio. Da mesma forma, a reputação de uma agência não garante equivalência entre o texto da alegação e a reportagem que a discute. Deve-se esclarecer qual unidade recebeu o rótulo: alegação, artigo jornalístico ou reportagem de checagem.

**Limite da evidência:** os logs, métricas, exemplos e regras de adaptação da tentativa com FACTCKBR não estão preservados nas fontes disponíveis deste relatório. Portanto, não se atribuem mensagens de erro, taxas de acerto ou uma causa quantitativamente demonstrada a essa tentativa.

Para uma retomada, é necessário preservar a versão do corpus, o esquema original, o mapeamento de rótulos, a distribuição de comprimentos e os exemplos avaliados. Uma avaliação de alegações curtas exige um protocolo específico: adaptação e validação no domínio de alegações, ou recuperação de artigos correspondentes com proveniência e revisão. Ela deve ser apresentada separadamente da classificação de notícias completas.

## 7. Notebook 03 — Holdout e avaliação externa congelada

O notebook 03 carrega o bundle treinado e valida os hashes. Não retreina o modelo na avaliação atual.

| Avaliação do stacking completo | N | Acurácia | F1-macro | Recall Fake | Recall Real |
|---|---:|---:|---:|---:|---:|
| Holdout interno | 2.384 | 0,9933 | 0,9933 | 99,08% | 99,58% |
| Externo extrativo auditado | 46.229 | 0,7871 | 0,7675 | 53,90% | 99,92% |

No externo, a matriz de confusão, com linhas de referência e colunas de previsão na ordem Fake/Real, é:

| Referência / Previsão | Fake | Real |
|---|---:|---:|
| Fake | 11.481 | 9.819 |
| Real | 21 | 24.908 |

O principal problema foi classificar notícias rotuladas Fake como Real. A AUC da classe Real foi 0,9873, mas uma AUC elevada não garante desempenho adequado no limiar empregado. O corpus é derivado e sua independência não foi confirmada.

No histórico versionado antigo, o notebook denominado de validação externa treinava um LinearSVC dentro de um split do corpus abstrativo. Isso constitui avaliação interna desse corpus, não validação externa de um modelo previamente congelado. A estrutura atual distingue treinamento, holdout e robustez externa.

## 8. Notebook 04 — Diagnóstico, erros e perturbação

O diagnóstico inclui quantidades por classe, baseline majoritário, métricas globais, margens, grupos por fonte/categoria/comprimento, autoria conhecida ou ausente, pistas editoriais e exemplos de erros. Foram investigados falsos negativos Fake, seus escores e sua concentração por fonte, categoria e ano extraído do campo Data.

Nos 9.819 erros Fake como Real, a mediana da margem Real foi **1,7857**. Aproximadamente **27,49%** tinham margem até 1, **55,54%** até 2 e **3,37%** acima de 5. Esses resultados descrevem os escores dos erros; não justificam selecionar novo limiar usando esse teste já observado.

A perturbação esvazia título, notícia ou autor em uma amostra determinística, mantendo o modelo congelado. Essa análise mede sensibilidade das entradas, **não é ablação com retreinamento** e não estabelece o efeito causal de retirar uma feature do treinamento.

### 8.1 Auditoria das dobras

A reconstrução de índices auditou 35 separações: cinco externas, 25 internas das externas e cinco internas do ajuste final. Não encontrou IDs ou textos normalizados exatos compartilhados nas dobras. Entretanto, **todas as 35 separações compartilhavam grupos de similaridade, fontes e autores**.

Entre treino e holdout foram registrados zero IDs, zero textos exatos e zero grupos de similaridade compartilhados, mas **219 autores e 13 fontes em comum**. O protocolo evita contaminação direta por linhas e ajuste global do TF-IDF nas dobras; não demonstra ausência total de dependência entre eventos, fontes ou autoria.

### 8.2 Comparação dos modelos congelados no externo

| Modelo | F1-macro | Recall Fake | Fake como Real |
|---|---:|---:|---:|
| Random Forest V2 | 0,8651 | 72,23% | 5.914 |
| MultinomialNB V1, protocolo distinto | 0,8429 | 67,87% | 6.844 |
| Stacking V2 completo | 0,7675 | 53,90% | 9.819 |
| LinearSVC V2 | 0,7046 | 43,37% | 12.063 |
| Regressão Logística V2 | 0,6704 | 38,01% | 13.204 |

A Random Forest teve melhor resultado que o stacking completo nesse corpus. O NB V1 é uma comparação complementar com outro protocolo. Esses achados não equivalem à seleção validada de um novo campeão externo independente.

## 9. Notebook 05 — Ablação com retreinamento

O notebook 05 é um **Retraining-Based Ablation Study**. Cada variante, incluindo a referência original, é clonada sem estado aprendido e retreinada nas mesmas partições, hiperparâmetros e dobras externas/internas.

Nas variantes sem autoria, o transformer de autor é removido. Nas variantes sem título, o processamento textual recebe apenas Notícia. O TF-IDF, os classificadores base e o meta-modelo são ajustados novamente. O esquema legado aceita as três colunas, mas isso não significa que campos excluídos participem das features.

### 9.1 Resultados da execução completa anterior

| Features | F1-macro OOF | F1-macro holdout | F1-macro externo | Recall Fake externo | Fake como Real externo |
|---|---:|---:|---:|---:|---:|
| Título + Notícia + Autor | 0,9900 | 0,9933 | 0,7675 | 53,90% | 9.819 |
| Título + Notícia | 0,9763 | 0,9778 | **0,8437** | **68,42%** | **6.726** |
| Somente Notícia | 0,9501 | 0,9539 | 0,6474 | 34,91% | 13.865 |
| Notícia + Autor | 0,9882 | 0,9912 | 0,6664 | 37,35% | 13.344 |

O modelo completo venceu internamente; sem autoria venceu externamente entre as quatro variantes. Retirar autoria mantendo título e notícia produziu **+0,0761 de F1-macro externo**, **+14,52 pontos percentuais de recall Fake** e **3.093 erros Fake como Real a menos**. A retirada elevou Real como Fake de 21 para 149, portanto há uma troca entre os tipos de erro.

O padrão é compatível com dependência de autoria/fonte no desenvolvimento, mas não demonstra a causa da mudança de desempenho. A combinação Título + Autor sem corpo não foi testada. Foram exportados deltas pareados por dobra e resultados por autoria conhecida, inédita ou ausente. A seleção da variante sem autoria foi realizada após observar o externo; seus testes seguintes devem ser descritos como exploratórios até confirmação independente.

**Estado da reexecução na elaboração:** o CSV incremental continha apenas as variantes original e sem autoria em OOF. Finalizar a reexecução é necessário antes de confirmar que todos os arquivos da ablação descrevem o mesmo conjunto de modelos.

## 10. V3 — Inferência por URL

O notebook `v3_teste_inferencia_dados_nao_vistos.ipynb` implementa configuração, carregamento do artefato sem autoria, entrada de URL, download/extração com Trafilatura, validação dos campos, preparação, inferência congelada, interpretação, ficha factual e registro. Uma seção final reúne métricas e gráficos de colunas.

A notícia é identificada como **candidata para o teste de inferência**. Há triagem de coincidências de URL, título, corpo e texto combinado contra corpus original, treino, holdout e externo anterior. Ausência de coincidência exata não garante novidade por evento.

O campo Autor vazio permanece apenas para compatibilidade com o validador legado. O notebook inspeciona os transformers e confirma que as bases usam exclusivamente Titulo + Noticia. Não executa fit.

### 10.1 Execução documentada

Fonte: execução `20261009_153902_781f6e01`, registrada em 9 de outubro de 2026, às 15h39, horário de São Paulo.

URL: https://veja.abril.com.br/politica/michelle-bolsonaro-chega-ao-fim-da-campanha-com-apenas-uma-doacao/

| Métrica | Resultado registrado |
|---|---|
| Classe prevista | 1 — REAL |
| Score Real / margem assinada | 8,602148 |
| Limiar de decisão | 0 |
| Distância ao limiar em unidades de score | 8,602148 |
| Probabilidade estimada Real | 99,9816%, sem calibração externa validada |
| Tempo de extração | 0,400465 s |
| Tempo total das chamadas de inferência | 0,162183 s |
| Texto analisado | 298 palavras; 1.843 caracteres |
| Cobertura TF-IDF por base | 57,04% dos termos/ngramas; 172 atributos ativos |
| Completude estrutural | Título e corpo recuperados; validações aprovadas |
| Verificações técnicas | Concluídas; artefato preservado |
| Referência factual | Não estabelecida por evidências registradas |
| Concordância factual | Não avaliável sem referência |

### 10.2 Como a decisão é produzida

Título e notícia são combinados e transformados pelos TF-IDFs aprendidos por cada base. Regressão Logística, Random Forest e LinearSVC produzem as entradas para a Regressão Logística final. Na decisão binária padrão, um score positivo favorece Real e um score negativo favorece Fake; o limiar é zero.

O score é uma margem assinada. Seu valor absoluto descreve intensidade em unidades de score, não confiança factual ou distância geométrica normalizada. `predict_proba` fornece estimativas do meta-modelo, sem calibração externa validada. A probabilidade próxima de 100% não confirma a veracidade da reportagem.

Os tempos são medições de uma execução, não médias de benchmark. O tempo total inclui chamadas separadas de predict, decision_function e predict_proba, com repetição de processamento. A extração estrutural não confirma automaticamente a recuperação de todo o conteúdo editorial.

**Conclusão V3:** o pipeline funcionou tecnicamente nessa execução e retornou Real. Os campos de evidência da ficha factual estão vazios e o rótulo de referência é nulo; não se pode afirmar acerto ou erro factual. A retirada de avisos visuais de revisão não substitui o preenchimento das evidências.

## 11. Problemas técnicos encontrados e correções

| Problema | Natureza | Correção ou tratamento |
|---|---|---|
| Caminhos absolutos e pasta V2 renomeada | Organização/portabilidade | Busca da raiz e atualização das referências para v2_validacao_externa |
| Rótulos externos invertidos | Semântica dos dados | Mapeamento antes das métricas; manutenção do padrão 0 Fake / 1 Real |
| Arrays tratados como Iterable; quantis e chaves genéricas | Diagnósticos estáticos | Conversões explícitas de arrays e valores |
| Atributos de estimadores não reconhecidos pelo Pylance | Tipagem | Acesso à configuração por get_params e uso de métodos disponíveis |
| NAType em chamadas split | Tipagem de pandas | Conversão para str antes da contagem de palavras |
| include_groups não reconhecido | Compatibilidade de API/tipagem | Amostragem por grupo com concatenação explícita |
| Variável reutilizada como lista e DataFrame; Never não iterável | Inferência de tipos | Variáveis distintas e tipos explícitos para tabelas |
| Trafilatura ausente | Ambiente do kernel | Instalação com o Python do kernel e invalidação dos caches de importação |
| Bloqueio EXTRACAO_REVISADA | Controle de fluxo introduzido no notebook | Remoção do bloqueio, preservando validações estruturais |
| probabilidades não definida | Execução fora de ordem ou estado antigo | Definição na inferência e recuperação dos resultados nas células posteriores |
| Diagnósticos antigos após edição | Estado de execução do editor/kernel | Reabrir a versão salva e executar em ordem, sem interromper treinamento em andamento |
| Caracteres substituídos por ? na ablação | Codificação | Correção de acentos e cedilhas; gravação UTF-8 |

Correções de tipagem não constituem novos resultados experimentais. Modificar o código em disco não atualiza automaticamente as células já executadas em um kernel. A recuperação de resultados na V3 pode repetir operações quando variáveis estão ausentes; os tempos devem ser interpretados para a execução efetivamente registrada.

## 12. Reprodutibilidade e publicação

As versões registradas na execução V2 são Python 3.14.3, scikit-learn 1.9.0, pandas 3.0.5 e NumPy 2.4.3. A execução V3 registra Trafilatura 2.3.1. São versões extraídas dos artefatos locais, não uma afirmação sobre versões atuais disponíveis para instalação.

Para reproduzir, é necessário manter os notebooks e auxiliares, datasets ou instruções de obtenção, modelo utilizado, partições e manifesto compatíveis e relatórios lidos como entradas. O 04/05 depende de mapa de dobras, métricas OOF, previsões OOF, métricas de holdout e previsões/metricas do subconjunto externo.

Caches, ambientes virtuais, modelos legados sem uso e gráficos duplicados não precisam ser publicados para executar os experimentos atuais. Não se deve ignorar indiscriminadamente reports, pois parte de seus arquivos é dependência operacional. Os gráficos da V3 foram configurados para aparecer apenas no próprio notebook; arquivos de imagens de execuções antigas não são novas dependências.

Ainda é necessário preparar um arquivo de dependências e executar o projeto a partir de uma cópia limpa para comprovar reprodutibilidade fora da máquina de desenvolvimento. Não houve publicação no GitHub durante a elaboração deste relatório.

## 13. Conclusões e próximos passos

1. O stacking completo teve o melhor F1-macro OOF da V2, mas sofreu perda de recall Fake no corpus externo derivado.
2. A Random Forest individual superou o stacking completo nesse corpus; o ensemble não apresenta superioridade externa uniforme.
3. A execução completa anterior da ablação favoreceu Título + Notícia sem autoria no externo, enquanto a autoria favoreceu resultados internos. O achado indica uma hipótese de dependência do domínio, sem provar causalidade.
4. A tentativa FACTCKBR, conforme relato, apresentou mudança de distribuição entre artigos completos e alegações curtas. Deve receber protocolo próprio e documentação recuperável antes de integrar conclusões quantitativas.
5. A V3 demonstrou funcionamento operacional em uma URL e retornou Real. Não demonstrou veracidade factual ou generalização a partir de um único exemplo.
6. A ausência de duplicatas exatas não elimina compartilhamento de fontes, autores ou eventos. A revisão manual integral não foi concluída no manifesto analisado.

Prioridades: concluir a reexecução da ablação; congelar e identificar cada execução por hashes; estabelecer referências factuais documentadas para testes de inferência; avaliar por fonte/evento/período quando houver metadados adequados; confirmar a variante escolhida em outro conjunto independente; e só então discutir calibração ou limiares sem usar o teste final para seleção.

## 14. Fontes locais

- `notebooks/v1_baseline/`: notebooks e baseline_features.py/evaluation_plots.py.
- `reports/v1_texto_autor_exploratorio/benchmark.csv`.
- `reports/auditoria_checagem/resumo_triagem_2026-10-08.json`.
- `notebooks/v2_validacao_externa/`: notebooks 01–05, pipeline_utils.py e ablation_utils.py.
- `notebooks/v2_validacao_externa/data/processed/texto_autor_exploratorio_v4/manifest.json`.
- `reports/v2_stacking_texto_autor_exploratorio/comparacao_modelos.csv` e `execucao.json`.
- `reports/v2_stacking_texto_autor_exploratorio/avaliacao/holdout_metrics.json`.
- `reports/v2_stacking_texto_autor_exploratorio/avaliacao/fakerecogna_extrativo/metricas.json`.
- `reports/v2_stacking_texto_autor_exploratorio/avaliacao/fakerecogna_extrativo/investigacao/`: comparacao_modelos_congelados.csv, auditoria_protocolo.json e resumo_pontuacoes.json.
- `reports/v2_stacking_texto_autor_exploratorio/ablacao/interpretacao.md` e `conclusao_ablacao.md`, referentes à execução completa anterior.
- `reports/v3_teste_inferencia_dados_nao_vistos/20261009_153902_781f6e01/`: registro.json, metricas_conclusao.csv e verificacao_factual.json.
- Histórico versionado: commit `5b30dba`, antigo notebook 03 de v2_pipeline_linearsvc.
- FACTCKBR: esclarecimento textual fornecido pela responsável pelo projeto nesta conversa; sem logs ou resultados quantitativos disponíveis.
