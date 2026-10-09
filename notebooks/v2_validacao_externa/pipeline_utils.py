"""Contrato compartilhado pelos notebooks v2 e pelo aplicativo."""
from pathlib import Path
from typing import Any
import hashlib
import html
import re
import unicodedata
import pandas as pd

SCHEMA_VERSION = "v2_texto_autor_exploratorio_4"
INPUT_POLICY = "base_original_sem_revisao_manual_integral"
FEATURE_COLUMNS = ["Titulo", "Noticia", "Autor"]
TARGET_COLUMN = "Classe"
TRAIN_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
PROCESSED_COLUMNS = ["id_original"] + FEATURE_COLUMNS + ["texto", "label"]
EXTERNAL_COLUMNS = FEATURE_COLUMNS + ["Label"]
LABEL_NAMES = {0: "Fake", 1: "Real"}
MODEL_RELATIVE = "models/v2_stacking_texto_autor_exploratorio/stacking_bundle.joblib"

def project_root(start=None):
    base = Path(start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "data/FakeRecogna.xlsx").is_file() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Raiz do projeto não encontrada (data/FakeRecogna.xlsx).")

def normalize_text(text):
    text = unicodedata.normalize("NFC", html.unescape(str(text))).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compose_text(frame, title="Titulo", body="Noticia"):
    validate_schema(frame, [title, body], context="composição textual")
    return (frame[title].fillna("").astype(str).str.strip() + " " +
            frame[body].fillna("").astype(str).str.strip()).str.strip()

def validate_labels(values):
    labels = pd.to_numeric(values, errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Rótulos devem ser 0 = Fake ou 1 = Real, sem nulos.")
    return labels.astype(int)

def sha256_file(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def load_bundle(file) -> dict[str, Any]:
    import joblib
    if not Path(file).is_file():
        raise FileNotFoundError("Bundle stacking ausente. Execute o treinamento completo do notebook 02 antes do 03.")
    bundle = joblib.load(file)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Artefato incompatível. Execute os notebooks 01 e 02 corrigidos.")
    if bundle.get("label_names") != LABEL_NAMES:
        raise ValueError("Mapeamento de rótulos incompatível.")
    if bundle.get("manifest", {}).get("input_policy") != INPUT_POLICY:
        raise ValueError("Artefato com política de entrada incompatível. Reexecute 01 e 02.")
    model = bundle["pipeline"]
    if list(model.classes_) != [0, 1] or bundle.get("input_columns") != FEATURE_COLUMNS:
        raise ValueError("Pipeline incompatível com o contrato de texto e classes.")
    if bundle.get("model_type") != "stacking":
        raise ValueError("Execute o notebook 02 de stacking para gerar o novo bundle.")
    from sklearn.ensemble import StackingClassifier
    stacking = model.named_steps["clf"]
    if not isinstance(stacking, StackingClassifier) or stacking.passthrough:
        raise ValueError("Estrutura de stacking incompatível.")
    if set(stacking.named_estimators_) != {"regressao_logistica", "random_forest", "svc"}:
        raise ValueError("Bases do stacking incompatíveis.")
    for base in stacking.estimators_:
        vectorizer, encoder = fitted_base_features(base)
        if vectorizer.preprocessor is not normalize_text or encoder.handle_unknown != "ignore":
            raise ValueError("Pré-processamento incompatível em uma base do stacking.")
    return bundle

# Triagem conservadora: sinais editoriais, não vereditos automáticos.
CHECKING_PATTERN = re.compile(
    r"\b(?:boatos?|fals[oa]s?|fake(?:\s+news)?|enganos[oa]s?|checagem|"
    r"desmentid[oa]s?|montagem|e[ -]?farsas|boatos\.org|comprova)\b", re.IGNORECASE
)
REVIEW_POLICY = "v2_texto_autor_pre_checagem_1"

def checking_signals(text):
    return sorted(set(m.group(0).casefold() for m in CHECKING_PATTERN.finditer(str(text))))

def prepare_review(frame):
    result = frame[["Titulo", "Noticia", "Classe"]].copy()
    result["label_original"] = validate_labels(result.pop("Classe"))
    result["id_original"] = frame.index
    result["texto_original"] = compose_text(result)
    result["sha256_texto_original"] = result.texto_original.map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    result["pistas_editoriais"] = result.texto_original.map(lambda text: " | ".join(checking_signals(text)))
    for column in ["decisao", "titulo_aprovado", "noticia_aprovada", "justificativa", "revisor"]:
        result[column] = ""
    return result

def apply_review(frame, review):
    required = {"id_original", "sha256_texto_original", "label_original", "decisao",
                "titulo_aprovado", "noticia_aprovada", "justificativa", "revisor"}
    if not required.issubset(review.columns):
        raise ValueError(f"Ficha incompleta: faltam {required - set(review.columns)}")
    if review.id_original.isna().any() or review.id_original.duplicated().any():
        raise ValueError("IDs ausentes ou repetidos na revisão.")
    expected = prepare_review(frame).set_index("id_original")
    review = review.set_index("id_original").fillna("")
    if set(review.index) != set(expected.index):
        raise ValueError("A revisão deve cobrir todos os registros elegíveis desta base.")
    review = review.loc[expected.index].copy()
    if not review.sha256_texto_original.eq(expected.sha256_texto_original).all():
        raise ValueError("Revisão desatualizada: texto original foi alterado.")
    reviewed_labels = validate_labels(review.label_original)
    if not reviewed_labels.eq(expected.label_original).all():
        raise ValueError("A revisão não pode inverter ou alterar os rótulos originais.")
    review["decisao"] = review.decisao.astype(str).str.strip().str.casefold()
    if not review.decisao.isin(["aprovar", "excluir"]).all():
        raise ValueError("Revisão pendente: marque aprovar ou excluir em todas as linhas.")
    for column in ["justificativa", "revisor"]:
        if review[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Preencha {column} em todas as decisões.")
    approved = review.loc[review.decisao.eq("aprovar")].copy()
    if approved.noticia_aprovada.astype(str).str.strip().eq("").any():
        raise ValueError("Toda aprovação exige o corpo/alegação original revisado.")
    approved["Titulo"] = approved.titulo_aprovado
    approved["Noticia"] = approved.noticia_aprovada
    approved["texto"] = compose_text(approved)
    approved["label"] = reviewed_labels.loc[approved.index]
    approved["pistas_residuais"] = approved.texto.map(checking_signals)
    # Pistas residuais são reportadas: citações legítimas não são removidas cegamente.
    return approved.reset_index()


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def validate_schema(frame, required_columns=None, context="entrada"):
    required = FEATURE_COLUMNS if required_columns is None else list(required_columns)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context}: esperado DataFrame com {required}.")
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{context}: nomes de colunas duplicados: {duplicates}.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{context}: colunas obrigatórias ausentes: {missing}. "
            f"Esperadas: {required}. Recebidas: {frame.columns.tolist()}. "
            "Use os nomes exatos, incluindo maiúsculas e acentos."
        )
    return frame

class InputSchemaValidator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_schema(X, context="treinamento")
        return self

    def transform(self, X):
        validate_schema(X, context="transformação/inferência")
        return X.loc[:, FEATURE_COLUMNS].copy()

def prepare_authors(frame):
    validate_schema(frame, ["Autor"], context="codificação de autor")
    authors = frame["Autor"].fillna("").astype(str).map(lambda x: " ".join(x.casefold().split()))
    date_like = authors.str.contains(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", regex=True)
    authors = authors.mask(authors.eq("") | date_like, "autor_desconhecido")
    return authors.to_numpy().reshape(-1, 1)

def make_features(max_features=20000):
    columns = ColumnTransformer([
        ("texto", Pipeline([
            ("combinar", FunctionTransformer(compose_text, validate=False)),
            ("tfidf", TfidfVectorizer(preprocessor=normalize_text, lowercase=False,
                                     max_features=max_features, ngram_range=(1, 2))),
        ]), ["Titulo", "Noticia"]),
        ("autor", Pipeline([
            ("normalizar", FunctionTransformer(prepare_authors, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), ["Autor"]),
    ], sparse_threshold=1.0)
    return Pipeline([("schema", InputSchemaValidator()), ("columns", columns)])

def prepare_author_review(frame):
    validate_schema(frame, FEATURE_COLUMNS + ["Classe"], context="base original")
    result = prepare_review(frame)
    result["Autor"] = frame["Autor"]
    result["sha256_autor_original"] = frame["Autor"].fillna("").astype(str).map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    result["autor_original_aprovado"] = ""
    result["proveniencia_alegacao"] = ""
    return result

def apply_author_review(frame, review):
    validate_schema(review, ["id_original", "sha256_autor_original", "autor_original_aprovado",
                             "proveniencia_alegacao", "decisao"], context="ficha texto e autor")
    expected = prepare_author_review(frame).set_index("id_original")
    if review.id_original.isna().any() or review.id_original.duplicated().any():
        raise ValueError("IDs ausentes ou repetidos na revisão.")
    indexed = review.set_index("id_original").fillna("")
    if set(indexed.index) != set(expected.index):
        raise ValueError("A revisão deve cobrir todos os registros elegíveis.")
    indexed = indexed.loc[expected.index]
    if not indexed.sha256_autor_original.eq(expected.sha256_autor_original).all():
        raise ValueError("Autor original alterado: revisão desatualizada.")
    approved = apply_review(frame, review)
    selected = indexed.loc[approved.id_original]
    if selected.proveniencia_alegacao.astype(str).str.strip().eq("").any():
        raise ValueError("Toda aprovação exige proveniencia_alegacao.")
    # Autor da alegação original; nunca assume a autoria do checador.
    approved["Autor"] = selected.autor_original_aprovado.to_numpy()
    return approved


def ensure_preprocessed_data(root):
    """Gera os artefatos do 01 somente quando estão ausentes, sem treinar modelos."""
    import json
    data_dir = Path(root) / "notebooks/v2_validacao_externa/data/processed/texto_autor_exploratorio_v4"
    expected = [data_dir / name for name in ["manifest.json", "train.csv", "test_holdout.csv"]]
    if all(path.is_file() for path in expected):
        return
    notebook_path = Path(root) / "notebooks/v2_validacao_externa/01_preproc_eda.ipynb"
    print("Artefatos ausentes: executando o pré-processamento do notebook 01. Isso pode levar alguns minutos.")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__preprocessamento_v2__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                exec(compile(source, f"{notebook_path.name}:celula_{index + 1}", "exec"), namespace)
            except Exception as erro:
                raise RuntimeError(
                    f"Pré-processamento 01 falhou na célula {index + 1}: {erro}. "
                    "Corrija essa etapa antes do treinamento."
                ) from erro
    if not all(path.is_file() for path in expected):
        raise RuntimeError("O notebook 01 terminou sem gerar todos os artefatos esperados.")


def load_manifest(data_dir) -> dict[str, Any]:
    import json
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Partições não preparadas. Execute o notebook 01 ou o início do 02.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION or
            manifest.get("input_columns") != FEATURE_COLUMNS or
            manifest.get("input_policy") != INPUT_POLICY or
            manifest.get("label_names") != {str(k): v for k, v in LABEL_NAMES.items()}):
        raise ValueError("Manifesto incompatível. Execute novamente o notebook 01.")
    if set(manifest.get("files", {})) != {"train.csv", "test_holdout.csv"}:
        raise ValueError("Manifesto incompleto. Execute novamente o notebook 01.")
    return manifest


def read_partition(path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False,
                        dtype={"Titulo": str, "Noticia": str, "Autor": str, "texto": str})
    validate_schema(frame, PROCESSED_COLUMNS, context=Path(path).name)
    if frame.empty or frame.id_original.astype(str).str.strip().eq("").any() or frame.id_original.duplicated().any():
        raise ValueError(f"{Path(path).name}: partição vazia ou IDs ausentes/duplicados.")
    frame["label"] = validate_labels(frame["label"])
    if frame.Noticia.str.strip().eq("").any() or frame.texto.map(normalize_text).eq("").any():
        raise ValueError(f"{Path(path).name}: corpo ou texto vazio.")
    if not compose_text(frame).map(normalize_text).eq(frame.texto.map(normalize_text)).all():
        raise ValueError(f"{Path(path).name}: texto diverge de Titulo + Noticia.")
    if frame.texto.map(normalize_text).duplicated().any():
        raise ValueError(f"{Path(path).name}: duplicatas normalizadas.")
    return frame


def fitted_base_features(base) -> tuple[TfidfVectorizer, OneHotEncoder]:
    """Retorna componentes ajustados com tipos explícitos para diagnóstico."""
    from typing import cast
    base = cast(Pipeline, base)
    features = cast(Pipeline, base.named_steps["features"])
    columns = cast(ColumnTransformer, features.named_steps["columns"])
    text = cast(Pipeline, columns.named_transformers_["texto"])
    author = cast(Pipeline, columns.named_transformers_["autor"])
    return (cast(TfidfVectorizer, text.named_steps["tfidf"]),
            cast(OneHotEncoder, author.named_steps["onehot"]))
