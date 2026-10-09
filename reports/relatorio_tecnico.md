# Relatório Técnico — Pipeline de Classificação de Fake News

**Programa:** Residência em Inteligência Artificial — Universidade de Brasília (UnB)  
**Projeto:** Identificador de Fake News / Hub de Verificação de Notícias  
**Data de referência:** 9 de outubro de 2026  
**Escopo:** preparação de dados, modelagem, avaliação, explicabilidade e operação de modelos de processamento de linguagem natural (NLP).

## Resumo do projeto

O projeto Identificador de Fake News / Hub de Verificação de Notícias investiga a classificação de notícias em português para apoiar a triagem de conteúdos potencialmente desinformativos e a revisão humana. O desenvolvimento documentado abrange preparação e auditoria de dados, modelos baseline com TF-IDF, um Stacking Ensemble com Regressão Logística, Random Forest e LinearSVC, avaliação interna e externa, diagnóstico de erros, ablação com retreinamento e inferência por URL com extração via Trafilatura. A rastreabilidade é apoiada por manifestos, hashes, modelos salvos e registros dos experimentos.

Este relatório distingue implementação observável, evolução proposta e resultados hipotéticos. Os resultados quantitativos apresentados para V1, V2 e V3 foram extraídos dos registros locais e identificados por etapa e execução. Versionamento com DVC, API FastAPI e análise de tom com mDeBERTa em modo zero-shot são mencionados no documento de referência fornecido, mas sua implementação não foi confirmada neste workspace. Fine-tuning supervisionado de transformers para classificação de Fake News, integração do treinamento com MLflow e implantação em AWS/GCP são possibilidades de evolução arquitetural, não resultados demonstrados pelos experimentos aqui descritos. Eventuais comparações hipotéticas entre famílias de modelos devem ser identificadas como simulações e separadas das métricas efetivamente medidas.

A principal restrição científica é que padrões linguísticos associados ao rótulo FAKE não comprovam a falsidade de uma alegação. A solução deve comunicar essa diferença e complementar a classificação textual com evidências externas e revisão especializada. O funcionamento técnico do pipeline e a intensidade de seus escores não substituem a verificação factual independente.

## 1. Contexto, objetivos e estado do desenvolvimento

O problema foi formulado como classificação supervisionada binária, com o contrato de rótulos **0 = FAKE e 1 = REAL**. A classe positiva deve ser explicitada por métrica: o F1-macro considera ambas as classes; na V2, a ROC-AUC e a margem de decisão usam REAL como referência positiva. Avaliações de FAKE como classe de interesse precisam aplicar a transformação correspondente. Esse contrato deve acompanhar treinamento, avaliação, explicações e resposta da API.

Os objetivos técnicos são estabelecer uma baseline reproduzível, investigar ganhos de representação contextual, tornar predições auditáveis e organizar o ciclo de vida dos modelos. A qualidade precisa ser examinada junto com custo, latência, cobertura e consequências dos erros.

A tabela a seguir incorpora o estado descrito no documento de referência fornecido. Os arquivos ml/train.py, ml/inference.py, data.dvc e docs/ENTREGA_EVOLUCAO_2026-10-08.md não foram localizados no workspace dos notebooks V1–V3; portanto, as observações sobre DVC, calibração, transformer, MLflow e API são informações relatadas nesse documento, sem confirmação nesta inspeção. A evolução abaixo não deve ser confundida com os componentes efetivamente executados nos notebooks.
| Componente | Situação observada | Evolução discutida |
| --- | --- | --- |
| Dados e DVC | Diretório data/ referenciado por data.dvc; configuração remota consultada sem remoto definido | Armazenamento remoto, pipeline declarativo e vínculo entre dados e experimentos |
| Modelo linear | TF-IDF, LinearSVC e calibração; candidatos separados do modelo servido | Seleção de hiperparâmetros e validação independente recente |
| Transformer | mDeBERTa NLI opcional, zero-shot, para análise auxiliar de tom | Fine-tuning binário e comparação controlada com a baseline |
| MLflow | Banco local com experimento Default, sem runs, métricas ou modelos registrados | Tracking integrado e governança no Model Registry |
| API | FastAPI, endpoints legados e v2, explicações e protótipo de verificação por evidências | Homologação, observabilidade e implantação em nuvem |
| Retreinamento | Geração de candidatos; retreino automático com votos não revisados desativado | Incorporação de dados revisados e promoção controlada |

Segundo o documento de referência, essas constatações se baseiam no código e nos artefatos locais, especialmente [README](README.md), [treinamento](ml/train.py), [inferência](ml/inference.py) e [entrega técnica](docs/ENTREGA_EVOLUCAO_2026-10-08.md). Documentos anteriores contêm afirmações mais amplas de prontidão; elas não substituem evidências de execução, avaliação e implantação.


No desenvolvimento efetivamente registrado neste projeto, a V1 estabelece os baselines; a V2 combina Regressão Logística, Random Forest e LinearSVC em um Stacking Ensemble, avalia holdout e corpus externo e investiga os erros. O notebook 05 retreina quatro conjuntos de features mantendo o protocolo de desenvolvimento. A V3 carrega a variante Título + Notícia sem autoria e realiza inferência congelada por URL. Os objetivos são comparar desempenho interno e externo, investigar dependência de autoria e documentar decisões com rastreabilidade. A comparação com FACTCKBR foi relatada como mudança de distribuição entre notícias completas e alegações curtas, não como falha de código.

## 2. Preparação, coleta e versionamento dos dados

### 2.1. Origem e coleta

O acervo local identificado como FakeRecogna contém 11.902 registros, originalmente distribuídos em 5.951 exemplos FAKE e 5.951 REAL, conforme a informação do documento de referência e sua [investigação do corpus](docs/INVESTIGACAO_MODELO_2026-10-08.md). O arquivo fake_recogna_limpo.csv fornece os campos utilizados no treinamento: título, notícia, classe, data e URL. A presença do arquivo não comprova sua procedência completa; a documentação de origem deve acompanhar cada versão.

No documento de referência, a aplicação utiliza httpx e Trafilatura para obter o conteúdo textual e remover elementos de navegação. No notebook V3 deste projeto, o download é realizado com requests e a extração com Trafilatura. Esse fluxo atende à inferência e à consulta de evidências. Sua saída não deve ingressar automaticamente no conjunto supervisionado: cada incorporação exige verificação da origem, anotação e revisão do rótulo.


Na V2 deste workspace, a entrada principal é data/FakeRecogna.xlsx, com Titulo, Noticia, Autor e Classe. Após o pré-processamento, o manifesto registra 9.518 exemplos de treino e 2.384 de holdout, vinculados por hashes SHA-256. O corpus externo data/fakerecogna_extrativo.csv contém 52.800 registros originais, dos quais 46.229 foram retidos na auditoria. Sua coluna Label utiliza a codificação inversa à do treinamento e é convertida antes da avaliação. Esses artefatos não devem ser confundidos com as partições e sistemas de versionamento descritos no documento de referência.

Para ampliar o corpus, o processo proposto registra:

- URL, domínio, título e corpo original, além da versão extraída e normalizada;
- datas de publicação, atualização, coleta e referência factual;
- identificador do artigo, hash do conteúdo, idioma e grupo de duplicação ou narrativa;
- rótulo, evidências, responsável pela anotação e resolução de divergências;
- condições de uso da fonte e limitações da coleta.

A unidade de anotação deve ser definida previamente. Uma notícia pode combinar alegações corretas e incorretas; nesse caso, é necessário explicitar a regra de agregação para o rótulo do artigo ou manter a avaliação por alegação.

### 2.2. Limpeza e preservação da informação

O pré-processamento dos experimentos atuais está implementado em `notebooks/v1_baseline/baseline_features.py` e `notebooks/v2_validacao_externa/pipeline_utils.py`. A entrada é validada pelos nomes exatos das colunas; título e corpo são combinados, e a autoria é tratada separadamente como variável categórica quando participa do experimento. A V3 reutiliza o processamento do pipeline retreinado e congelado, mantendo a mesma representação textual utilizada no treinamento.

A normalização textual aplica:

- normalização Unicode NFC e conversão para minúsculas, preservando letras acentuadas;
- substituição de URLs por `urltoken`;
- remoção de marcação HTML e substituição de caracteres fora de letras, números, sublinhado e espaços por separadores;
- redução de espaços repetidos e remoção de espaços nas extremidades;
- na V2, decodificação de entidades HTML com `html.unescape` antes das demais transformações.

As funções atuais não removem stopwords nem executam stemming ou lematização. A negação “não” permanece na normalização, assim como dígitos. Entretanto, pontuação, símbolos e separadores são removidos: datas como `09/10/2026`, valores decimais e percentuais perdem sua forma original. Portanto, não se deve afirmar que datas, unidades monetárias ou percentuais são integralmente preservados. O próprio tokenizador do TF-IDF também determina quais sequências se tornam atributos. Essa simplificação deve ser considerada ao interpretar alegações factuais que dependam de números ou relações entre valores.

Na composição textual, valores ausentes de título são convertidos em texto vazio. Isso não autoriza aceitar corpo ausente: os carregadores e validadores verificam notícias vazias, texto normalizado vazio, rótulos inválidos e inconsistências entre o texto composto e as colunas de origem. A autoria é normalizada por `casefold` e espaços; valores vazios ou semelhantes a datas são representados por `autor_desconhecido`. O one-hot ignora categorias inéditas na transformação. Na variante sem autoria, usada pela V3, esse transformer é excluído, embora o validador legado ainda aceite uma coluna Autor vazia.

Os dados de entrada, os campos textuais e o identificador original são mantidos separados das representações derivadas. As partições processadas contêm `Titulo`, `Noticia`, `Autor`, `texto`, `label` e `id_original`, e seus hashes são registrados no manifesto. A auditoria acrescenta sinais editoriais para inspeção, sem removê-los automaticamente ou usá-los como confirmação de falsidade. Expressões presentes em reportagens de checagem podem revelar o processo editorial e devem ser examinadas no contexto.

A existência de versões anteriores do dataset limpo exige cuidado: normalização não recupera conteúdo que já tenha sido eliminado em uma etapa anterior. A preservação dos arquivos de origem e a documentação das transformações permitem identificar essa limitação. Na V3, o HTML capturado e os campos extraídos com Trafilatura são registrados separadamente da entrada enviada ao pipeline. Recuperar título e corpo e passar nas validações estruturais não comprova, por si só, que todo o conteúdo editorial foi extraído.

No treinamento V2, cada base ajusta seu TF-IDF dentro das dobras correspondentes; na inferência, os vocabulários permanecem congelados. Essa separação mantém consistência entre preparação, treinamento e operação e evita ajustar o vocabulário com os dados de avaliação. Mudanças futuras na política de limpeza devem ser versionadas e acompanhadas de retreinamento e nova avaliação, em vez de aplicadas apenas à inferência de um modelo já treinado.

### 2.3. Deduplicação e divisão dos dados

A deduplicação utiliza como chave o texto normalizado formado por **Título + Notícia**. São excluídos registros sem corpo ou rótulo, notícias com conteúdo vazio e textos normalizados vazios. Antes de remover repetições, o código verifica se uma mesma chave textual apresenta rótulos diferentes; caso encontre conflito, interrompe a preparação para exigir correção. As duplicatas exatas com rótulo consistente são reduzidas a um registro, preservando o identificador original para rastreabilidade.

No pré-processamento registrado da V2, o arquivo de origem tinha **11.903 linhas**, das quais **11.902 foram retidas**. O resumo registra uma remoção agregada por ausência, vazio ou duplicação, sem discriminar nesse arquivo qual dessas condições determinou a exclusão. Assim, 11.902 corresponde ao acervo elegível após a preparação, não ao número bruto de linhas lidas nessa execução.

Para identificar textos semelhantes, o notebook 01 da V2 aplica `HashingVectorizer` com **262.144 atributos**, unigramas e bigramas, sem sinais alternados e com normalização L2. A busca usa distância cosseno e até **seis vizinhos por registro**, incluindo o próprio registro, que é descartado na comparação. Pares com similaridade igual ou superior a **0,90** são unidos em componentes conectados por uma estrutura de agrupamento. Foram registrados **164 pares candidatos** no corpus preparado, dos quais **133** estavam no treino. Essa busca limitada a vizinhos é uma triagem aproximada; não assegura localizar todas as paráfrases ou notícias sobre o mesmo evento.

A configuração escolhida foi `SPLIT_MODE = "similaridade"`. Os componentes são separados com `GroupShuffleSplit`, uma divisão, `test_size=0.2` e semente **42**. A fração de 20% é aplicada aos grupos, não garante exatamente 20% dos artigos e não realiza estratificação explícita por classe. O código exige que ambas as classes estejam presentes em treino e holdout e bloqueia textos normalizados ou grupos de similaridade compartilhados entre essas partições.

| Partição V2 | Registros | Papel no experimento |
|---|---:|---|
| Treino | 9.518 | Validação cruzada de desenvolvimento e ajuste final |
| Holdout | 2.384 | Avaliação do modelo congelado no notebook 03 |
| Total preparado | 11.902 | Corpus elegível após filtragem e deduplicação |

Não existe uma partição separada de calibração no protocolo atual da V2. A separação cronológica em treino, calibração e teste e os números anteriormente apresentados para outro candidato não descrevem esta execução.

O notebook também avalia a viabilidade estrutural de alternativas, sem usar suas métricas preditivas para escolher a separação:

| Alternativa | Situação registrada |
|---|---|
| Similaridade | Escolhida: 9.518 exemplos de treino e 2.384 de holdout |
| Fonte | Viável estruturalmente: 9.289 de treino e 2.613 de holdout; não foi a configuração escolhida |
| Evento | Indisponível: metadados de evento ausentes ou inválidos |
| Período | Indisponível: não havia datas válidas para todos os registros |

Na alternativa por fonte, componentes semelhantes que compartilham a mesma fonte são unidos antes da divisão. Na alternativa por período, quando viável, o código reserva registros mais recentes e retira do treino grupos semelhantes presentes no teste. Esses mecanismos existem no código, mas **não justificam afirmar que a execução escolhida foi temporal ou independente por fonte/evento**.

Depois da divisão, os notebooks 02 e 05 utilizam **cinco dobras externas estratificadas**, com embaralhamento e semente 42, sobre o treino. O stacking gera ainda previsões out-of-fold em **cinco dobras internas** para ajustar o meta-modelo. Cada base aprende seu próprio TF-IDF dentro das dobras correspondentes. Na ablação, todas as variantes reutilizam os mesmos índices externos e o mesmo protocolo; o ajuste final é feito apenas no treino completo.

A auditoria das 35 separações externas/internas registrou zero IDs e zero textos normalizados exatos em comum, mas grupos semelhantes, fontes e autores compartilhados em todas elas. Entre treino e holdout houve zero IDs, zero textos exatos e zero grupos semelhantes compartilhados, porém **13 fontes e 219 autores em comum**. As dobras de desenvolvimento são estratificadas por classe, não agrupadas por similaridade, fonte ou evento. Portanto, os controles reduzem contaminação direta entre linhas, mas não demonstram ausência total de dependência ou vazamento semântico.

Na V1, o baseline usa divisão estratificada 80%/20%, com `train_test_split` e semente 42; seu protocolo não deve ser confundido com a divisão por grupos da V2. Na avaliação externa V2, o corpus extrativo passa por filtros adicionais de coincidência de URL, título, texto e similaridade, mantendo os mesmos registros retidos para a comparação das variantes. Isso não certifica independência de origem ou evento, pois se trata de um corpus derivado.

As partições V2 são salvas como `train.csv` e `test_holdout.csv`, acompanhadas de `manifest.json`, em `notebooks/v2_validacao_externa/data/processed/texto_autor_exploratorio_v4/`. O manifesto registra parâmetros, quantidades e hashes SHA-256; o mapa de dobras e os relatórios de auditoria permitem conferir a reprodução do protocolo. Holdout e externo já observados não devem orientar ajustes posteriores sem um novo teste independente.

Fontes: `01_preproc_eda.ipynb`, `02_treinamento_ensemble_stacking.ipynb`, `05_ablacao.ipynb`, `manifest.json`, `preproc_eda/resumo_auditoria.json`, `preproc_eda/avaliacao_estrutural_separacoes.csv` e `investigacao/auditoria_protocolo.json`, nos diretórios da V2 e seus relatórios.

### 2.4. Versionamento com DVC

Não foram encontrados `data.dvc`, `dvc.yaml`, `dvc.lock` ou configuração `.dvc` no workspace atual dos notebooks V1–V3. Portanto, não há evidência local de um pipeline DVC executado, de remoto configurado ou de recuperação de dados por DVC. As afirmações do documento de referência sobre esse mecanismo não descrevem uma implementação verificada neste projeto.

A rastreabilidade efetivamente implementada utiliza Git para o código e artefatos locais com manifestos e hashes SHA-256. Na V2, `manifest.json` registra esquema, campos de entrada, rótulos, política de preparação, semente, configuração de separação, quantidades e hashes do dataset original, `train.csv` e `test_holdout.csv`. Os notebooks validam esses dados antes de treinar ou avaliar. Bundles de modelos incluem metadados do treinamento, e avaliações registram hashes do modelo e do dataset. Na V3, cada execução cria um identificador próprio e registra captura, extração, entrada, triagem, versões e resultado.

Esses controles identificam alterações e incompatibilidades, mas não substituem armazenamento remoto, histórico completo de grandes artefatos ou uma reprodução testada a partir de uma cópia limpa. A reexecução de alguns notebooks sobrescreve relatórios em caminhos fixos; o histórico completo deve ser preservado antes da execução seguinte.

DVC é uma possibilidade de evolução para versionar dados e modelos volumosos. Um fluxo futuro pode declarar preparação, divisão, treinamento, ablação e avaliação como estágios, mantendo a correspondência entre commit, parâmetros, hashes e identificador de execução. A configuração de remoto e sua recuperação devem ser validadas antes de apresentar esse fluxo como operacional. Não há partição separada de calibração na V2 atual, nem fine-tuning de transformer implementado nos notebooks examinados.

## 3. Pipeline de treinamento e configuração dos modelos

O desenvolvimento implementado utiliza representações TF-IDF e classificadores tradicionais. A V1 estabelece comparações baseline; a V2 treina e avalia um Stacking Ensemble; o notebook 05 retreina variantes de features; a V3 executa inferência com uma dessas variantes congelada. As etapas são descritas abaixo conforme o código e os resultados locais, sem atribuir implantação de API, MLflow ou treinamento de transformers à execução observada.

### 3.1. Fluxo geral

~~~mermaid
flowchart TD
    A[FakeRecogna original] --> B[Validação e normalização de título e notícia]
    B --> C[Deduplicação e agrupamento por similaridade]
    C --> D[Treino 9518 e holdout 2384 com manifesto]
    D --> E[Cinco dobras externas sobre o treino]
    E --> F[Stacking com cinco dobras internas e TF-IDF por base]
    F --> G[Previsões OOF e ajuste final no treino]
    G --> H[Avaliação congelada no holdout e corpus externo auditado]
    H --> I[Diagnóstico de erros e perturbação de entradas]
    I --> J[Ablação com retreinamento de quatro configurações]
    J --> K[Variante sem autoria congelada]
    K --> L[URL e extração Trafilatura na V3]
    L --> M[Validação e inferência FAKE ou REAL]
    M --> N[Registro, gráficos e referência factual separada]
~~~

A representação original combina Título + Notícia com autoria categórica one-hot. Cada base da V2 contém seu próprio pipeline de features e ajusta vocabulário, IDF e categorias dentro das dobras. As bases são Regressão Logística, Random Forest e LinearSVC; o meta-modelo é outra Regressão Logística. `passthrough=False` impede acrescentar as features originais diretamente à entrada do meta-modelo.

O holdout não participa do ajuste do notebook 02. O corpus externo tem seus rótulos convertidos para **0 = Fake e 1 = Real** e é avaliado com o modelo congelado. Sua natureza derivada e a observação posterior dos resultados tornam a seleção da variante sem autoria exploratória. O notebook 05 mantém as mesmas partições, hiperparâmetros e dobras entre variantes, retreinando também a referência original. O notebook 04 apenas esvazia campos com o modelo congelado na análise de perturbação; os dois procedimentos são distintos.

Na V3, não há fit ou calibração: são carregados os TF-IDFs e o stacking retreinados sem autoria. A notícia passa por extração e triagem de coincidências antes da previsão. O registro separa resultado estatístico e veredito factual. O diagrama representa a sequência metodológica e as dependências principais; não afirma que existe um orquestrador automático de todas as etapas.

### 3.2. Baseline: TF-IDF e LinearSVC

O TF-IDF transforma texto em vetores esparsos, ponderando a frequência de termos no documento pela frequência inversa no corpus de treino. O LinearSVC aprende uma função linear de separação entre classes nesse espaço. Sua margem de decisão não é uma probabilidade calibrada.

No benchmark atual da V1, `baseline_features.make_features(max_features=10000)` combina TF-IDF de Título + Notícia e one-hot de Autor. O vetorizador utiliza o normalizador compartilhado da V1 e os demais padrões não explicitamente alterados, incluindo unigramas. O classificador é `LinearSVC(random_state=42)`. A separação é estratificada, com 80%/20% e semente 42. O LinearSVC alcançou **F1-macro de 0,9861 e acurácia de 98,61%** na execução registrada, com 33 erros. Esse resultado pertence ao protocolo V1 e não deve ser tratado como comparação controlada com todas as avaliações da V2.

Na V2, o LinearSVC é uma base do stacking e também é avaliado isoladamente. A configuração explícita é TF-IDF com até **20.000 atributos**, unigramas e bigramas, normalização compartilhada da V2, autoria one-hot quando ativa e `LinearSVC(dual=True, max_iter=10000, random_state=42)`. O valor de C permanece no padrão 1,0; não há busca de hiperparâmetros nessa etapa. Cada base recebe seu próprio processamento, sem TF-IDF global ajustado antes das dobras.

~~~python
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from pipeline_utils import make_features

base_svc = Pipeline([
    ("features", make_features(max_features=20000)),
    ("clf", LinearSVC(dual=True, max_iter=10000, random_state=42)),
])
# A avaliação clona e ajusta esta base dentro de cada dobra.
# O ajuste final usa somente o conjunto de treino preparado.
~~~

A base LinearSVC V2 apresentou **F1-macro OOF de 0,9878** e, na avaliação externa extrativa congelada, **0,7046**, com recall Fake de **43,37%**. A diferença de domínio e as dependências entre fontes/eventos devem ser consideradas na interpretação.

Não há `CalibratedClassifierCV` ou uma partição de calibração independente nos pipelines atuais descritos. O LinearSVC isolado fornece margens; no stacking, essas margens podem alimentar a Regressão Logística final. O `predict_proba` da V3 é fornecido pelo meta-modelo, não por uma calibração externa do LinearSVC. Sua disponibilidade não demonstra calibração factual. Seleção de C, avaliação por grupos/período e calibração com dados separados são possibilidades futuras, que exigem protocolo próprio e novo teste independente.

### 3.3. Transformers multilingues: zero-shot e fine-tuning
**Estado no projeto:** esta seção apresenta fundamentação e proposta experimental. Os modelos efetivamente treinados nos notebooks atuais utilizam TF-IDF e classificadores tradicionais, incluindo o stacking. Não há resultado de zero-shot ou fine-tuning de mDeBERTa registrado para comparação com essas execuções. Os checkpoints e a configuração abaixo não integram o pipeline servido pela V3.

O mDeBERTa-v3 oferece representações contextuais multilingues que motivam sua investigação para relações semânticas menos dependentes da ocorrência exata de palavras. A família DeBERTa-v3 utiliza pré-treinamento com detecção de tokens substituídos, conforme o [artigo original](https://arxiv.org/abs/2111.09543). Essa propriedade fundamenta uma hipótese de ganho; não demonstra superioridade no corpus do projeto.

O checkpoint atualmente referenciado, MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, foi ajustado para inferência de linguagem natural (NLI) e permite classificação zero-shot pela comparação entre texto e hipóteses. No documento de referência fornecido, esse checkpoint é descrito como componente auxiliar para classificar tom neutro ou opinativo/sensacionalista. Esse uso não foi confirmado no código ou nos artefatos dos notebooks V1–V3 examinados. Essa tarefa difere da classificação supervisionada FAKE/REAL. A avaliação publicada no [model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) não comprova desempenho em notícias brasileiras.

Para o fine-tuning proposto, o protocolo é inicializar o encoder multilingue e uma cabeça binária, treinar com artigos anotados e avaliar em português. Uma opção é partir de microsoft/mdeberta-v3-base; outra é comparar a inicialização pelo checkpoint NLI, substituindo sua cabeça de três classes. Origem e revisão dos pesos devem ser fixadas antes da execução.

Configuração inicial de referência proposta, sem execução comparativa documentada neste projeto:

~~~yaml
transformer:
  checkpoint: microsoft/mdeberta-v3-base
  revision: <commit-do-checkpoint-a-fixar>
  num_labels: 2
  label_mapping: {0: FAKE, 1: REAL}
  max_length: 512
  padding: dynamic
  learning_rate: 2.0e-5
  epochs_max: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  optimizer: AdamW
  weight_decay: 0.01
  warmup_ratio: 0.10
  early_stopping_patience: 2
  selection_metric: macro_f1
  evaluation_strategy: epoch
  seed: 42
  precision_initial: fp32
~~~

O bloco define intenções experimentais; não é um arquivo diretamente consumido pelo Trainer. Os nomes de parâmetros, como evaluation_strategy, devem ser adaptados à versão de Transformers fixada para o experimento. A acumulação corresponde a 32 artigos por passo de atualização com um dispositivo. A implementação deve mapear os campos para a versão fixada da biblioteca, tokenizar título e corpo, calcular a perda e selecionar o checkpoint somente pela validação. O fluxo de ajuste supervisionado está documentado pela [Hugging Face](https://huggingface.co/docs/transformers/tasks/sequence_classification).

Para artigos longos, devem ser comparados truncamento e janelas sobrepostas, registrando a agregação das predições por artigo. Todas as janelas de uma notícia pertencem à mesma partição. Precisão reduzida é uma otimização posterior, condicionada à estabilidade numérica do checkpoint e do hardware. Calibração e escolha dos limiares também precisam usar dados separados da seleção do checkpoint e do teste final.

## 4. Critérios de seleção da abordagem

### 4.1. Complementaridade e decisão arquitetural

A evolução do projeto distingue os experimentos já realizados da próxima abordagem proposta. A **V1** estabelece baselines; a **V2** utiliza Stacking Ensemble e ablação com retreinamento; a **V3** testa a inferência por URL com o stacking congelado sem autoria. A **V4 proposta** acrescenta **mDeBERTa-v3 supervisionado**, ajustado especificamente para classificação binária de notícias em português: **0 = FAKE e 1 = REAL**. Não há treinamento ou métricas de V4 documentados nos artefatos examinados.

| Critério | Baseline TF-IDF + LinearSVC — V1 | Stacking TF-IDF — V2 e inferência V3 | mDeBERTa-v3 supervisionado — V4 proposta |
|---|---|---|---|
| Representação | Padrões lexicais | TF-IDF por base; autoria quando ativa | Representação contextual multilingue de título e corpo |
| Decisão | Classificador linear | Regressão Logística combina saídas das bases | Cabeça binária ajustada com rótulos FAKE/REAL |
| Estado | Benchmark executado | Treinamento, avaliação, ablação e inferência registrados | Fine-tuning e avaliação comparativa a executar |
| Hipótese investigada | Referência lexical de menor complexidade | Complementaridade das bases e efeito das features | Captura de relações contextuais além da correspondência lexical |
| Seleção | Métricas do protocolo baseline | Métricas internas e externas, com limites de independência | Validação própria e confirmação em teste independente |
| Explicabilidade | Contribuições à margem por atributo | Análise por base, meta-modelo, perturbação e ablação | Atribuições e perturbações a validar experimentalmente |
| Custo e latência | Medidos no benchmark local | Registrados por treinamento e inferência | A medir no hardware e configuração definidos |
| Evidência de veracidade | Exige fontes externas | Exige fontes externas | Também exige fontes externas |

A V4 não corresponde ao uso zero-shot do checkpoint NLI para identificar tom neutro, opinativo ou sensacionalista. No fine-tuning supervisionado, os pesos e a cabeça de classificação são ajustados com exemplos anotados para a tarefa FAKE/REAL. A análise de tom, caso implementada posteriormente, permanece um componente auxiliar: sensacionalismo não equivale a falsidade, e tom neutro não confirma veracidade.

A comparação controlada deve usar a mesma unidade de anotação, o mesmo contrato de rótulos e os mesmos artigos de teste para os candidatos comparados. O transformer não utiliza TF-IDF; utiliza o tokenizer correspondente ao checkpoint escolhido. Para uma comparação sem autoria, título e corpo devem ser a entrada comum. Caso autoria seja investigada na V4, essa alteração exige experimento separado. As partições precisam impedir que janelas do mesmo artigo sejam distribuídas entre treino e teste.

O protocolo deve fixar a revisão do checkpoint, a representação de artigos longos, os hiperparâmetros e a regra de agregação antes da avaliação final. Seleção de checkpoint e hiperparâmetros ocorre na validação de desenvolvimento. Calibração e escolha de limiar, quando realizadas, exigem dados apropriados e não devem usar o teste final. Como holdout e corpus externo da V2 já foram observados, seus resultados podem servir à comparação exploratória, mas a confirmação da V4 exige um novo teste independente ou protocolo prospectivo equivalente.

Os critérios de decisão incluem **F1-macro, precisão e recall por classe, erros FAKE como REAL e REAL como FAKE, estabilidade por fonte/evento/período, cobertura, tempo de treinamento, latência, memória e custo operacional**. Devem ser documentadas as incertezas e diferenças pareadas quando houver desenho adequado. O objetivo é investigar ganho preditivo e operacional, não presumir superioridade do transformer por sua arquitetura.

A execução completa anterior da ablação favoreceu **Título + Notícia sem autoria** entre as variantes de stacking no corpus externo observado. Entretanto, a **Random Forest individual da V2 apresentou F1-macro externo superior ao stacking completo**. Por isso, uma futura comparação V4 deve incluir tanto o stacking sem autoria quanto baselines relevantes e especificar suas features e protocolos; não basta compará-la apenas com o pior candidato externo.

A complementaridade arquitetural pode ser investigada posteriormente por uma cascata ou combinação de modelos. Encaminhamento por incerteza, pesos e limiares devem ser definidos em validação e avaliados como um sistema completo. Não existe, nos notebooks examinados, um ensemble que combine probabilidades do stacking e do mDeBERTa supervisionado. Uma explicação produzida pelo modelo linear não explica automaticamente a decisão do transformer ou do stacking.

A promoção de V4 depende de evidência de melhoria no protocolo estabelecido e de um custo operacional aceitável. Se não houver ganho consistente, os candidatos tradicionais permanecem referências válidas. Nenhuma dessas arquiteturas, isoladamente, substitui a verificação factual independente.

### 4.2. Explainable AI — XAI

A explicabilidade procura compreender **quais padrões e componentes influenciam a classificação**. No projeto, a interpretação combina margens de decisão, inspeção de erros, resultados por subgrupo, perturbação das entradas e ablação com retreinamento. Essas análises descrevem o comportamento do modelo; não estabelecem a veracidade de uma notícia ou a causa factual de uma alegação.

| Abordagem | Evidência ou uso no projeto | Limite de interpretação |
|---|---|---|
| Margem e limiar | V2/V3 exibem score de decisão e distância ao limiar | Intensidade do sinal não é certeza factual ou probabilidade calibrada |
| Diagnóstico de erros | Notebook 04 analisa erros por fonte, categoria, autoria, comprimento, período e margem | Associações por grupo não demonstram causalidade |
| Perturbação das entradas | Notebook 04 esvazia título, notícia ou autor no modelo congelado | Mede sensibilidade local; não equivale a treinar sem a feature |
| Ablação com retreinamento | Notebook 05 compara quatro conjuntos de features com mesmo protocolo | Mede diferenças entre variantes; não isola causas semânticas ou sociais |
| Cobertura do vocabulário | V3 mede termos/ngramas conhecidos e atributos TF-IDF ativos por base | Cobertura lexical não comprova compreensão ou acerto |
| Atribuições por termo | Possíveis em modelos lineares com pesos e features ajustados | Não foram confirmadas como rotina implementada de XAI nos notebooks atuais |
| Atribuições do transformer | Proposta para mDeBERTa supervisionado V4 | Exigem implementação e avaliação de fidelidade |

**Modelos lineares.** Em um classificador linear, o score pode ser decomposto como `f(x) = b + Σ(w_j × x_j)`. Cada produto corresponde à contribuição de um atributo vetorizado. Uma análise por termo exige recuperar o vocabulário correto, pesos, intercepto e orientação das classes; quando houver autoria one-hot, suas contribuições também precisam ser identificadas. A soma descreve o score, não parcelas aditivas de uma probabilidade. Não foi localizado `ml/xai.py` no workspace examinado, portanto não se afirma a existência desse motor no projeto atual.

**Stacking.** Cada base aplica sua representação e produz saídas usadas pela Regressão Logística final. Na V2, `stack_method="auto"` pode fornecer probabilidades de algumas bases e margem do LinearSVC; essas entradas possuem significados e escalas diferentes. O meta-modelo pode ser examinado por suas entradas, coeficientes e intercepto, mas uma contribuição de termo de uma base não explica diretamente toda a decisão do ensemble. Seria necessário analisar também a transformação das saídas e sua combinação final. `passthrough=False` mantém as features originais fora da entrada direta do meta-modelo.

**Perturbação e retreinamento.** Esvaziar um campo na inferência mantém o modelo aprendido e pode alterar a normalização conjunta do TF-IDF ou produzir uma entrada fora da distribuição de treino. O notebook 04 registra essa sensibilidade sem atribuir efeito causal. Já o notebook 05 exclui os campos das features e reajusta TF-IDF, bases e meta-modelo. Na execução completa anterior, retirar autoria mantendo Título + Notícia elevou o F1-macro externo de 0,7675 para 0,8437, enquanto o modelo completo foi melhor internamente. O padrão é compatível com dependência de autoria/fonte no desenvolvimento, mas não demonstra que a autoria seja a causa única da perda de generalização.

**Inferência individual V3.** O relatório exibe classe prevista, margem assinada da classe REAL, limiar zero, distância absoluta ao limiar em unidades de score, probabilidades estimadas quando disponíveis e cobertura TF-IDF. Esses indicadores permitem compreender o sinal utilizado pelo classificador. A distância em score não é distância geométrica normalizada ao hiperplano; a probabilidade do meta-modelo não tem calibração externa validada. Uma margem elevada pode acompanhar uma previsão incorreta.

**V4 supervisionada proposta.** Para mDeBERTa, podem ser investigadas atribuições sobre embeddings e perturbações controladas de trechos. O procedimento deve documentar método, referência, tokens analisados, agregação de subpalavras e tratamento de janelas de artigos longos. Fidelidade e estabilidade precisam ser avaliadas: remover trechos relevantes deve produzir efeitos coerentes, comparados com perturbações de controle. Mapas de atenção isolados não devem ser tratados como justificativa suficiente ou como evidência factual. Ainda não há resultados dessas análises de V4 nos artefatos examinados.

A apresentação ao usuário deve separar **previsão estatística, explicação do componente responsável e evidências externas**. Título, corpo extraído, versão do modelo, hashes, score, limiar e referência factual devem acompanhar o registro. Fontes, nomes e termos associados ao rótulo podem expressar vieses editoriais do corpus; não são justificativas automáticas de falsidade. A interpretação deve corresponder ao modelo realmente executado e reconhecer as limitações da entrada e do protocolo de avaliação.

### 4.3. Infraestrutura AWS/GCP e endpoints FastAPI

**Estado atual:** os experimentos V1–V3 examinados são executados em notebooks e registram modelos e resultados no armazenamento local. Não foram confirmados uma API FastAPI, um Dockerfile operacional ou uma implantação AWS/GCP no workspace atual. A infraestrutura descrita nesta seção é uma proposta de evolução para disponibilizar a inferência e apoiar a futura V4 supervisionada.

A opção por nuvem decorre da necessidade de armazenar artefatos, executar treinamentos com recursos adequados e escalar a API independentemente dos experimentos. AWS e GCP constituem alternativas; utilizar ambas simultaneamente exige justificativa de custo, capacidade ou continuidade operacional.

| Camada | Opção AWS | Opção GCP | Critério técnico |
|---|---|---|---|
| Dados e artefatos | Amazon S3 | Google Cloud Storage | Objetos versionados, acesso controlado e eventual integração DVC |
| API conteinerizada em CPU | Amazon ECS/Fargate | Cloud Run | Escala do serviço e isolamento de dependências |
| Transformer acelerado — V4 proposta | Serviço ou instância com GPU, separado da API quando necessário | Cloud Run com GPU, quando compatível | Memória, latência, concorrência e disponibilidade regional |

As opções de objetos são descritas pela [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) e pelo [Google Cloud](https://docs.cloud.google.com/storage/docs/introduction). A documentação de [Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html) e de [GPU no Cloud Run](https://docs.cloud.google.com/run/docs/configuring/services/gpu) deve orientar uma decisão futura, com conferência das condições disponíveis no momento da implantação. A tabela não registra deploy realizado, custos contratados ou desempenho homologado. DVC também permanece uma possibilidade de evolução, conforme a seção 2.4.

Uma API FastAPI pode fornecer contratos de entrada e saída para disponibilizar o pipeline congelado. Os seguintes endpoints são preservados como **contratos propostos do documento de referência**, sem confirmação de implementação nos notebooks atuais:

- **POST /analisar-url:** análise textual de uma notícia extraída da URL;
- **POST /v2/analisar-url:** análise de URL com classificação e resultado de verificação apresentados separadamente;
- **POST /v2/verificar:** registro ou execução de um fluxo de alegações e evidências, quando esse componente for implementado;
- **GET /health:** estado de funcionamento do serviço;
- **GET /ready:** prontidão das dependências e do classificador carregado.

A nomenclatura `/v2` dos endpoints é uma versão do contrato de API e não implica, por si só, correspondência com a pasta experimental V2. A resposta de inferência deve identificar a versão do modelo, hashes, features efetivas, classe, score, limiar e probabilidade quando disponível. Um resultado de verificação factual exige evidências próprias; não deve ser preenchido automaticamente a partir da classificação FAKE/REAL.

O carregamento via `lifespan` é uma possibilidade para inicializar os modelos antes de atender requisições, conforme a [documentação do FastAPI](https://fastapi.tiangolo.com/advanced/events/). O planejamento deve considerar o carregamento por processo: múltiplos workers podem replicar os pesos na memória. Inferência custosa exige concorrência limitada e capacidade dimensionada; `async` não acelera, por si só, o cálculo do modelo.

O código de extração e inferência da V3 pode servir de referência para essa integração, mas a conversão de notebook em serviço exige contratos, tratamento de falhas de rede e extração, limites de requisição, observabilidade e testes. A validação deve distinguir corpo recuperado, completude editorial e ausência de coincidências com o corpus. Os tempos medidos em uma execução local não homologam latência ou capacidade de uma API em nuvem.

Não se atribui um comportamento específico a um Dockerfile que não foi localizado neste workspace. Uma imagem futura deve fixar dependências compatíveis com o artefato, disponibilizar os módulos auxiliares necessários à desserialização e verificar versões e hashes. Para V4, será necessário provisionar pesos, tokenizer e bibliotecas correspondentes. Downloads e instalação automática utilizados no notebook de desenvolvimento devem ser substituídos por provisionamento controlado para uma implantação reproduzível.

A adoção de nuvem permanece condicionada à escolha de uma arquitetura, teste de recuperação dos artefatos, validação funcional e medição de custo, latência e memória. Esta etapa não altera a conclusão científica: disponibilizar uma classificação por API não equivale a verificar a veracidade da notícia.

## 5. Rastreamento de experimentos com MLflow

### 5.1. Evidência disponível e integração proposta

No repositório atual, não foi localizado o arquivo `mlflow.db` nem uma integração de tracking no treinamento inspecionado. A informação do texto de referência sobre um banco contendo apenas o experimento `Default`, sem runs, parâmetros, métricas ou modelos registrados, não pôde ser confirmada neste ambiente. Portanto, os resultados existentes não devem ser apresentados como experimentos rastreados pelo MLflow.

O projeto já mantém registros locais em arquivos CSV e JSON, manifestos, predições, resultados dos notebooks e modelos serializados. Esses artefatos documentam as avaliações realizadas, mas ainda precisam ser associados a runs para compor um histórico centralizado no MLflow.

Na integração proposta, cada configuração recebe uma run. O registro deve distinguir a baseline V1, os candidatos e folds da V2, as variantes do estudo de ablação com retreinamento, a inferência com o modelo congelado na V3 e o fine-tuning supervisionado do mDeBERTa na V4, quando executado. São registrados parâmetros, métricas, hashes dos dados, commit, seed e artefatos. Esse desenho segue as funções de [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).

Comparações hipotéticas e resultados sintéticos devem receber identificação explícita, permanecer separados dos experimentos medidos e não ser elegíveis para promoção de modelos. Nenhum número hipotético representa uma métrica extraída de um banco MLflow do projeto.

| Registro proposto | Conteúdo |
| --- | --- |
| Parâmetros | Família, conjunto de features, hiperparâmetros, tokenizer quando aplicável, extensão do texto, calibração e limiares |
| Identificação | Commit Git, hashes dos dados e splits, seed, revisão dos pesos e versão DVC quando implementada |
| Métricas | Acurácia, precisão, recall, F1 por classe e macro, erros e cobertura, separados por conjunto avaliado |
| Artefatos | Matriz de confusão, predições por artigo, configuração, manifesto e pipeline treinado |
| Operação | Hardware, tempo de treino, tempo de extração, latência de inferência, memória e configuração de carga |

O registro da inferência V3 deve incluir URL, identificação do modelo congelado e condições de extração. Uma única previsão não fornece acurácia, precisão, recall ou F1; a concordância factual só pode ser registrada quando houver um veredito independente documentado.

Exemplo de instrumentação proposta, a incorporar ao treinamento após configurar o destino do tracking. Os parâmetros e tags devem conter valores serializáveis, e as métricas devem ser numéricas e calculadas pelo pipeline:

~~~python
import mlflow


def registrar_experimento(config, metricas, diretorio_artefatos):
    mlflow.set_experiment("residencia-unb-fake-news")
    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.log_params(config["params"])
        mlflow.set_tags(config["tags"])
        mlflow.log_metrics(metricas)
        mlflow.log_artifacts(str(diretorio_artefatos))
        return run.info.run_id
~~~

Esse exemplo registra metadados e arquivos; não implementa, por si só, o registro de um modelo no Model Registry nem a sua implantação.

### 5.2. Registro e promoção

Na integração proposta, o Model Registry deve associar versões aprovadas às runs e aos artefatos. Um alias como `champion` pode indicar a referência vigente, mas a implantação deve resolver e fixar a versão concreta. Aprovação requer teste independente, critérios por classe e avaliação operacional. O modelo anterior deve permanecer disponível para rollback. Essas capacidades são descritas pelo [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/).

Cada versão deve preservar o pipeline de pré-processamento, o classificador, o mapeamento de classes, os hashes e as dependências necessários para reproduzir a inferência. A promoção das variantes de ablação ou do mDeBERTa supervisionado da V4 depende de resultados medidos; os conjuntos já observados não constituem uma nova avaliação independente.

Não há evidência de registro ou promoção de modelos pelo MLflow no estado atual do projeto. Esta seção descreve a governança proposta para sua integração.

## 6. Métricas de desempenho e análise dos resultados

### 6.1. Definições e protocolo

O pipeline do projeto utiliza **0 = FAKE e 1 = REAL**. Para interpretar os erros nesta seção, FAKE é a classe de interesse, independentemente de seu código numérico. No CSV externo, a convenção é inversa (`Label 1 = Fake` e `Label 0 = Real`); os rótulos devem ser convertidos para a convenção do modelo antes do cálculo das métricas.

Considerando FAKE como classe positiva na interpretação:

- **TP:** notícia FAKE corretamente classificada como FAKE;
- **FP:** notícia REAL classificada como FAKE;
- **FN:** notícia FAKE classificada como REAL;
- **TN:** notícia REAL corretamente classificada como REAL.

| Métrica | Fórmula | Interpretação |
| --- | --- | --- |
| Acurácia | (TP + TN) / N | Fração total de acertos |
| Precisão FAKE | TP / (TP + FP) | Fração de alertas FAKE corretos |
| Recall FAKE | TP / (TP + FN) | Fração de notícias FAKE detectadas |
| F1 FAKE | 2TP / (2TP + FP + FN) | Equilíbrio entre precisão e recall para FAKE |
| F1 macro | (F1 FAKE + F1 REAL) / 2 | Média com peso igual para as duas classes |
| FPR | FP / (FP + TN) | Fração de notícias REAL indevidamente sinalizadas |
| FNR | FN / (FN + TP) | Fração de notícias FAKE não detectadas |

As matrizes de confusão devem informar a orientação: linhas representam a referência e colunas representam a previsão, na ordem FAKE e REAL. Nesse formato, a matriz corresponde a `[[TP, FN], [FP, TN]]`. Quando uma métrica binária tratar FAKE como positiva, é necessário especificar `pos_label=0`; o padrão numérico 1 corresponde a REAL no treinamento.

**Protocolo realizado.** A V1 utiliza divisão estratificada de desenvolvimento e teste, com seed 42. A V2 mantém partições fixas de treinamento e holdout, com agrupamento por similaridade textual para reduzir sobreposição entre essas partições. A comparação dos candidatos utiliza predições out-of-fold com cinco folds externos; o Stacking possui cinco folds internos para treinar o metaclassificador. Os vetorizadores e classificadores devem ser ajustados somente com os dados de treinamento de cada etapa. O holdout recebe o pipeline final congelado.

A validação externa aplica os modelos congelados ao subconjunto externo retido após a auditoria, com conversão dos rótulos e exclusões documentadas. O estudo de ablação retreina cada conjunto de features com o mesmo protocolo de desenvolvimento e avalia todas as variantes nos mesmos IDs de holdout e de dados externos, sem novas exclusões por variante. Assim, a comparação mede o efeito da configuração de features junto ao retreinamento.

**Critérios de interpretação.** F1 macro, recall FAKE, recall REAL e contagens dos dois tipos de erro devem ser analisados em conjunto. A elevada acurácia no holdout não garante generalização para outra fonte ou formato de texto. A avaliação externa e a experiência relatada com alegações curtas do FACTCKBR evidenciam a necessidade de examinar mudança de distribuição; não há métricas documentadas dessa tentativa com FACTCKBR para quantificar seu efeito.

Nas curvas ROC e no cálculo de AUC, é preciso alinhar a classe positiva ao score utilizado. No pipeline binário, a margem positiva aponta para REAL (classe 1); uma análise orientada a FAKE exige a inversão do score ou a seleção da probabilidade correspondente à classe 0. AUC avalia ordenação dos exemplos ao longo de limiares e não substitui recall e matriz de confusão no limiar operacional.

As probabilidades produzidas pelo metaclassificador do Stacking não foram demonstradas como calibradas em um conjunto independente. Curvas de confiabilidade e Brier score são avaliações complementares propostas. Desempenho por tema, fonte e período, intervalos de confiança e comparações pareadas também devem ser incorporados quando os metadados e agrupamentos necessários estiverem disponíveis.

**Limites do protocolo.** Holdout e conjunto externo já foram observados durante o desenvolvimento; as análises subsequentes são exploratórias e não constituem um novo teste independente para selecionar campeão ou limiar. Os folds internos e externos também não asseguram independência completa de fontes, autores ou narrativas. A comparação futura com o mDeBERTa supervisionado da V4 exige dados e critérios previamente definidos.

A V3 registra uma inferência individual: classe, score, margem, probabilidade quando disponível, tempos, completude da extração, tamanho do texto e cobertura do vocabulário. Esses indicadores avaliam a execução do pipeline. Uma única notícia não permite estimar acurácia, precisão, recall ou F1; o acerto factual depende de um veredito independente sustentado por evidências.
### 6.2. Comparação hipotética: baseline linear versus mDeBERTa-v3 supervisionado (V4)

**Todos os números desta subseção são hipotéticos e didáticos.** Não são resultados dos notebooks nem runs do MLflow. A V4 supervisionada é uma evolução proposta, sem avaliação comparativa executada documentada neste relatório. O cenário considera um mesmo teste fictício de 2.000 artigos, com 1.000 FAKE e 1.000 REAL, sem abstenção. Um limiar ilustrativo de probabilidade de 0,5 pressupõe uma saída probabilística; o LinearSVC puro utiliza margem de decisão e não oferece `predict_proba()` diretamente.

#### Matriz hipotética — TF-IDF + LinearSVC

Linhas representam a referência; colunas representam a previsão, na ordem FAKE e REAL.

| Referência / Predição | FAKE | REAL | Total |
| --- | ---: | ---: | ---: |
| FAKE | 900 (TP) | 100 (FN) | 1.000 |
| REAL | 80 (FP) | 920 (TN) | 1.000 |
| Total | 980 | 1.020 | 2.000 |

#### Matriz hipotética — mDeBERTa-v3 com fine-tuning

| Referência / Predição | FAKE | REAL | Total |
| --- | ---: | ---: | ---: |
| FAKE | 950 (TP) | 50 (FN) | 1.000 |
| REAL | 50 (FP) | 950 (TN) | 1.000 |
| Total | 1.000 | 1.000 | 2.000 |

#### Métricas derivadas das matrizes hipotéticas

| Métrica | TF-IDF + LinearSVC | mDeBERTa-v3 ajustado |
| --- | ---: | ---: |
| Acurácia | 91,00% | 95,00% |
| Precisão FAKE | 91,84% | 95,00% |
| Recall FAKE | 90,00% | 95,00% |
| F1 FAKE | 90,91% | 95,00% |
| F1 macro | 91,00% | 95,00% |
| FPR | 8,00% | 5,00% |
| FNR | 10,00% | 5,00% |

Nesse exemplo construído, a acurácia aumenta 4 pontos percentuais e o F1 FAKE aumenta aproximadamente 4,09 pontos. Os falsos negativos diminuem de 100 para 50 e os falsos positivos de 80 para 50. O erro total cai de 180 para 100 artigos, redução relativa de aproximadamente 44,44%.

Esses números ilustram como interpretar uma comparação; não demonstram superioridade do transformer. A hipótese de ganho com representações contextuais precisa ser testada com artigos anotados, partições controladas e análise de erros. Matrizes agregadas também não informam quais artigos receberam previsões divergentes e não permitem, sozinhas, afirmar significância estatística.

### 6.3. Resultados históricos efetivamente disponíveis

Os resultados abaixo correspondem aos registros locais do projeto V1–V3, resumidos em `reports/relatorio_tecnico_modelo_v1_v2_v3.md`. F1 macro é apresentado em escala de 0 a 1. Os protocolos da V1 e V2 diferem; seus valores não constituem uma comparação controlada da troca de algoritmo.

| Modelo no benchmark V1 | Acurácia | F1 macro |
| --- | ---: | ---: |
| LinearSVC | 0,9861 | 0,9861 |
| Regressão Logística | 0,9794 | 0,9794 |
| Random Forest | 0,9769 | 0,9769 |
| MultinomialNB | 0,9735 | 0,9735 |

O LinearSVC apresentou o melhor F1 macro no benchmark V1. Na comparação out-of-fold da V2, o Stacking apresentou o melhor resultado interno:

| Modelo V2 | F1 macro OOF | FAKE como REAL | REAL como FAKE |
| --- | ---: | ---: | ---: |
| Stacking | 0,9900 | 57 | 38 |
| LinearSVC | 0,9878 | 83 | 33 |
| Regressão Logística | 0,9770 | 181 | 38 |
| Random Forest | 0,9742 | 142 | 104 |

#### Holdout e validação externa do Stacking completo

| Avaliação | Artigos | Acurácia | F1 macro | Recall FAKE | Recall REAL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Holdout interno | 2.384 | 0,9933 | 0,9933 | 99,08% | 99,58% |
| Subconjunto externo auditado | 46.229 | 0,7871 | 0,7675 | 53,90% | 99,92% |

As matrizes registradas, com linhas de referência e colunas de previsão, são:

| Conjunto | FAKE → FAKE | FAKE → REAL | REAL → FAKE | REAL → REAL |
| --- | ---: | ---: | ---: | ---: |
| Holdout | 1.187 | 11 | 5 | 1.181 |
| Externo | 11.481 | 9.819 | 21 | 24.908 |

A queda externa concentra-se nas notícias FAKE classificadas como REAL. A AUC orientada à classe REAL foi aproximadamente 0,9873, mas a ordenação favorável dos scores não eliminou os erros no limiar utilizado. O conjunto externo é derivado e sua independência completa de fontes e eventos não foi confirmada.

#### Ablação com retreinamento

A execução completa anterior, documentada nos arquivos `interpretacao.md` e `conclusao_ablacao.md` em `reports/v2_stacking_texto_autor_exploratorio/ablacao/`, apresentou:

| Features | F1 macro OOF | F1 macro holdout | F1 macro externo | Recall FAKE externo |
| --- | ---: | ---: | ---: | ---: |
| Título + Notícia + Autor | 0,9900 | 0,9933 | 0,7675 | 53,90% |
| Título + Notícia, sem autoria | 0,9763 | 0,9778 | 0,8437 | 68,42% |
| Somente Notícia | 0,9501 | 0,9539 | 0,6474 | 34,91% |
| Notícia + Autor, sem título | 0,9882 | 0,9912 | 0,6664 | 37,35% |

**Título + Notícia, sem autoria, apresentou o melhor resultado externo entre as quatro variantes retreinadas.** Em relação ao modelo completo, o F1 macro externo aumentou aproximadamente 0,0761 e o recall FAKE aumentou 14,52 pontos percentuais. Os erros FAKE como REAL diminuíram de 9.819 para 6.726, enquanto REAL como FAKE aumentaram de 21 para 149. O resultado sugere dependência de padrões de autoria no corpus, mas não demonstra causalidade nem garante generalização para qualquer nova fonte.

A Random Forest individual da V2 alcançou F1 macro externo de 0,8651 e recall FAKE de 72,23%, superando o Stacking completo e a variante sem autoria nesse conjunto. Portanto, o melhor resultado interno do ensemble não implica superioridade externa uniforme. Esses resultados exploratórios não autorizam escolher um novo campeão após observar o teste.

#### Inferência individual da V3 e limites das evidências

A execução `20261009_153902_781f6e01` previu REAL para a URL analisada, com score 8,602148 e probabilidade estimada REAL de 99,9816%, sem calibração externa validada. Foram registrados aproximadamente 0,400465 s de extração e 0,162183 s nas chamadas de inferência, com 298 palavras, 1.843 caracteres e cobertura TF-IDF de 57,04% dos termos/ngramas. Esses valores descrevem uma execução, não um benchmark de carga nem a probabilidade de a notícia ser factualmente verdadeira.

O registro consultado não contém um veredito factual independente preenchido; assim, não é possível concluir que a previsão corresponde à realidade. Também não há métricas documentadas para a tentativa com FACTCKBR ou para a V4 supervisionada. Os resultados do candidato `candidate-20261008T195757552426Z` mencionados no texto de referência não foram confirmados nos artefatos deste repositório e não são utilizados como evidência nesta seção.

### 6.4. Decisão, prevalência e abstenção

O limiar deve refletir os custos dos erros e ser definido com dados de desenvolvimento apropriados antes da avaliação final. Falsos positivos sinalizam notícias REAL como FAKE; falsos negativos deixam notícias FAKE classificadas como REAL. Alterar o limiar após examinar o conjunto externo exige uma nova avaliação independente para confirmar o efeito.

Na inferência binária da V3, o score positivo aponta para REAL e a margem é interpretada em relação ao limiar zero. A distância ao limiar representa intensidade da decisão do classificador, não força de evidência factual. Probabilidades do Stacking também não devem ser comunicadas como certificação de veracidade.

Uma faixa de abstenção ou encaminhamento à revisão é uma evolução proposta, não uma funcionalidade já demonstrada neste experimento. Se implementada, deve registrar a cobertura, a taxa de revisão e o erro entre os casos decididos, preservando todos os artigos no balanço da avaliação. A acurácia seletiva não é diretamente comparável à acurácia de um modelo que classifica todos os exemplos.

A precisão depende da prevalência. Em um cenário hipotético com 10% de notícias FAKE, recall FAKE de 95% e FPR de 5%, a precisão FAKE seria `0,95 × 0,10 / (0,95 × 0,10 + 0,05 × 0,90) ≈ 67,86%`. Assim, desempenho em corpus balanceado não garante a mesma precisão no tráfego real. Essa conta pressupõe manutenção das taxas condicionais; mudança de distribuição pode alterá-las.

A decisão operacional deve combinar métricas por classe, avaliação de generalização, custo e revisão factual documentada. Classificação textual e verificação independente permanecem etapas distintas.
## 7. Principais desafios técnicos e estratégias de tratamento

### 7.1. Gerenciamento do ciclo de vida e MLOps

O desafio central é manter compatibilidade entre dados, pré-processamento, modelo e inferência. As versões V1, V2 e V3 precisam compartilhar contratos explícitos de campos, normalização e rótulos; aplicar uma representação nova a pesos antigos pode invalidar predições. A referência a uma limpeza denominada facts-v2 no texto fornecido não foi confirmada no repositório atual.

O treinamento comparativo, os modelos serializados, os manifestos, as partições persistidas e os hashes constituem avanços concretos. Permanecem como evolução a integração DVC/MLflow, a reprodução em ambiente limpo e critérios de promoção que combinem qualidade, custo e risco. Seeds reduzem variação, mas não garantem identidade numérica entre hardwares e runtimes.

O texto de referência menciona a desativação do retreino automático com votos, mas essa implementação não foi confirmada no projeto atual. Feedback comunitário deve alimentar uma fila de revisão, não substituir rótulos verificados. Após revisão, os dados podem gerar candidatos; a atualização do modelo servido depende de aprovação e possibilidade de rollback.

### 7.2. Integração entre serviços de nuvem

Uma arquitetura distribuída entre AWS e GCP introduz diferenças de identidade, permissões, armazenamento, rede e observabilidade. Transferências de corpus e pesos podem gerar custo e atraso; falhas parciais dificultam descobrir qual versão foi efetivamente usada.

O tratamento proposto envolve imagens portáveis, contratos HTTP versionados, hashes de artefatos, credenciais temporárias e rastreamento de requisições. Dados e inferência devem permanecer na mesma região e provedor quando isso atender aos requisitos. Treinamento em outro ambiente pode publicar um pacote imutável para implantação, evitando comunicação entre nuvens em cada predição.

Esses pontos são desafios de desenho e homologação; o repositório não demonstra operação multicloud validada.

### 7.3. Desbalanceamento e mudança de distribuição

Embora o corpus de desenvolvimento seja balanceado, o subconjunto externo avaliado contém 21.300 exemplos FAKE e 24.929 REAL. Não há evidência suficiente para afirmar o comportamento dos recortes temporais neste projeto. O tráfego pode variar por tema, fonte e período; também pode haver poucos exemplos representativos de certos grupos linguísticos.

As estratégias propostas incluem pesos de classe no LinearSVC, perda ponderada no transformer e amostragem somente no treino, comparadas com a baseline sem reponderação. O teste deve preservar a distribuição de interesse. Reamostrar antes da divisão produz risco de vazamento; interpolar vetores TF-IDF não garante exemplos linguisticamente válidos.

O monitoramento deve separar mudança dos dados de mudança da relação entre conteúdo e rótulo. Drift de entrada é um alerta; demonstrar degradação de classificação exige novos rótulos revisados. Não foi confirmada uma implementação de monitoramento operacional contínuo no repositório atual.


No caso do FACTCKBR, o problema relatado foi de adequação dos dados: o modelo aprendeu com notícias completas, enquanto o conjunto fornecia principalmente alegações curtas de fact-checking. Essa mudança de distribuição altera a extensão, a estrutura e a disponibilidade de título e autoria. O tratamento exige definir a unidade de classificação, auditar o mapeamento dos rótulos e avaliar um modelo treinado ou adaptado para alegações. Não há métricas locais documentadas dessa tentativa.

A ablação com retreinamento também revelou sensibilidade à autoria: remover esse campo e manter título e notícia melhorou o F1 macro externo, embora tenha reduzido o desempenho interno. Esse resultado motiva avaliações por fonte, autor e narrativa e impede tratar a acurácia interna como garantia de generalização.

### 7.4. Latência e memória do mDeBERTa-v3 supervisionado — V4 proposta

O transformer adiciona tokenização, cálculo neural e maior ocupação de memória. Extensão dos textos, número de hipóteses no zero-shot, batch, hardware e concorrência afetam o tempo de resposta. Inicialização dos pesos também pode comprometer disponibilidade durante a escala.

As medidas propostas são:

- provisionar pesos previamente, inicializar o modelo por processo e limitar concorrência;
- controlar tokens, usar padding dinâmico e avaliar janelas com agregação por artigo;
- testar batches e separar workers de inferência pesada da API quando necessário;
- investigar quantização, exportação e precisão reduzida, medindo regressão de qualidade;
- usar cache com chave de conteúdo, versão do modelo e pré-processamento, além de validade temporal;
- calcular explicações custosas sob demanda e avaliar cascata com revisão dos casos inconclusivos.

Não há benchmark comparável do mDeBERTa supervisionado que sustente uma latência específica neste relatório. Os tempos individuais registrados pela V3 correspondem ao pipeline TF-IDF e Stacking e não podem ser atribuídos ao transformer. A homologação deve medir p50, p95 e p99, throughput, memória e taxa de erro em carga. Tokenização, inferência, extração web e busca externa devem ter medições separadas. Execuções frias e aquecidas precisam ser identificadas.

### 7.5. Qualidade factual e avaliação contemporânea

Um detector textual pode aprender atalhos relacionados a fonte, época ou tema. Mudanças superficiais de estilo podem enganá-lo sem alterar a alegação. Essa limitação reforça a necessidade de verificação por evidências e de teste recente anotado independentemente. A V3 prevê o registro de verificação factual, mas os artefatos consultados não estabelecem um veredito independente para a notícia analisada.

A avaliação deve incluir paráfrases, negações, números alterados, textos formais falsos, notícias verdadeiras com linguagem emotiva e casos com evidência insuficiente. Reputação de domínio e tom neutro contextualizam a análise, mas não substituem a verificação de cada alegação.

## 8. Aprendizados e lições adquiridas

### 8.1. Melhores práticas de MLOps

As evidências do desenvolvimento sustentam as seguintes lições:

1. **Reprodutibilidade depende do conjunto completo de artefatos.** Código, dados, splits, pré-processamento, dependências e pesos precisam estar vinculados.
2. **Ferramentas instaladas não comprovam adoção operacional.** A presença ou menção a MLflow e DVC não demonstra rastreamento e recuperação completos; a integração dessas ferramentas não foi confirmada no repositório atual.
3. **Candidatos devem ser separados do modelo congelado usado na inferência.** Essa prática permite auditar alterações e avaliar regressões antes da promoção.
4. **Treino e inferência precisam compartilhar contratos.** Correções de limpeza exigem retreinamento ou manutenção explícita da representação legada.
5. **Feedback é uma fonte de investigação.** Revisão independente protege contra ruído, manipulação e contaminação do corpus.
6. **Ablação com retreinamento exige novos ajustes dos modelos.** Esvaziar campos de um modelo congelado mede perturbação, não o mesmo experimento realizado no notebook 05.
7. **Convenções de rótulos precisam ser auditadas.** O CSV externo exige inversão para a codificação do treinamento, enquanto o holdout mantém seus rótulos originais.

### 8.2. Pipelines escaláveis e escolha de modelos

A baseline linear estabelece uma referência de custo e desempenho que torna a avaliação de modelos maiores mais objetiva. A adoção de transformers deve decorrer de ganhos medidos no domínio de interesse, além dos resultados em benchmarks externos.

Pipelines modulares permitem evoluir extração, preparação, treino, avaliação e serving com responsabilidades claras. Escalabilidade também depende de limites de entrada, filas, cache, controle de concorrência e recuperação de falhas. A representação de texto deve preservar informação suficiente para a tarefa antes da busca por otimizações de hardware.

### 8.3. Impacto ético e social

Uma ferramenta de combate à desinformação pode ampliar a capacidade de triagem e apoiar educação midiática. Também pode produzir acusações indevidas, reforçar vieses do corpus ou induzir confiança excessiva em scores.

A prática responsável exige comunicar incerteza, apresentar fontes e permitir contestação. Avaliações por tema, período, fonte e variedade linguística ajudam a identificar danos assimétricos. Opinião, sátira, linguagem informal e sensacionalismo precisam ser tratados segundo critérios explícitos de anotação; não equivalem automaticamente a falsidade.

A revisão humana deve ser acionável e documentada, especialmente em casos de maior impacto. A utilidade social depende tanto da detecção de conteúdo problemático quanto da capacidade de preservar informação legítima e corrigir decisões equivocadas.

## 9. Conclusão e próximos passos

O projeto estabeleceu um fluxo de preparação de dados, comparação de classificadores, treinamento de Stacking, validação externa, ablação com retreinamento e inferência por URL com modelo congelado. Os resultados mostram elevada discriminação interna, mas também perda de generalização externa e limitações de independência de fontes e eventos.

O Stacking completo apresentou o melhor F1 macro OOF da V2 (0,9900) e F1 macro de 0,9933 no holdout, porém caiu para 0,7675 no subconjunto externo, com recall FAKE de 53,90%. Entre as quatro variantes da ablação, Título + Notícia sem autoria apresentou o melhor F1 macro externo (0,8437), sugerindo dependência de padrões de autoria. A Random Forest individual atingiu 0,8651 nesse conjunto, demonstrando que o ensemble não foi superior em todas as condições.

A V3 demonstrou a execução registrada de extração, preparação e inferência com a variante sem autoria. Sua previsão REAL e sua probabilidade estimada não comprovam a veracidade da notícia; o registro consultado não permite afirmar concordância com um veredito factual independente. A experiência relatada com FACTCKBR reforça que alegações curtas e notícias completas representam unidades de análise diferentes.

A comparação hipotética com mDeBERTa-v3 ajustado ilustra a interpretação de métricas, mas não autoriza sua promoção. Os próximos passos são integrar tracking com MLflow, implementar versionamento recuperável com DVC, reproduzir o pipeline em ambiente limpo, obter um corpus recente revisado e executar a V4 supervisionada sob protocolo controlado. A avaliação deve registrar desempenho por classe, independência dos grupos, calibração, latência e custo. Implantação em AWS ou GCP permanece uma proposta sujeita à homologação.

O produto deve apoiar a análise e comunicar os limites de cada saída. Classificação textual, análise de tom e verificação factual são tarefas distintas; a utilidade do sistema depende de evidências documentadas e da possibilidade de revisar suas decisões.