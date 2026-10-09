"""Features da v1: título + notícia (TF-IDF) e autor categórico."""
from pathlib import Path
import re
import hashlib
import warnings
import json
import unicodedata
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_COLUMNS = ["Titulo", "Noticia", "Autor"]
TARGET_COLUMN = "Classe"
TRAIN_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
LABEL_NAMES = {0: "Fake", 1: "Real"}


def validate_schema(frame, required_columns=None, context="entrada"):
    """Valida nomes exatos; não infere colunas por posição ou aliases."""
    required = FEATURE_COLUMNS if required_columns is None else list(required_columns)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context}: esperado pandas.DataFrame com colunas {required}.")
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
    """Executa a validação antes das features no treino e na inferência."""
    def fit(self, X, y=None):
        validate_schema(X, context="treinamento")
        return self

    def transform(self, X):
        validate_schema(X, context="transformação/inferência")
        return X.loc[:, FEATURE_COLUMNS].copy()

def project_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "data/FakeRecogna.xlsx").is_file():
            return candidate
    raise FileNotFoundError("Execute dentro do projeto com data/FakeRecogna.xlsx.")

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def combine_text(frame):
    validate_schema(frame, ["Titulo", "Noticia"], context="composição textual")
    return (frame["Titulo"].fillna("").astype(str) + " " +
            frame["Noticia"].fillna("").astype(str)).str.strip()

def prepare_authors(frame):
    validate_schema(frame, ["Autor"], context="codificação de autor")
    authors = frame["Autor"].fillna("").astype(str).map(lambda x: " ".join(x.casefold().split()))
    # Há datas no campo Autor da base. Não trate esses valores como identidades.
    date_like = authors.str.contains(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", regex=True)
    authors = authors.mask(authors.eq("") | date_like, "autor_desconhecido")
    return authors.to_numpy().reshape(-1, 1)

def make_features(max_features):
    columns = ColumnTransformer([
        ("texto", Pipeline([("combinar", FunctionTransformer(combine_text, validate=False)),
                            ("tfidf", TfidfVectorizer(preprocessor=normalize_text,
                                                      lowercase=False, max_features=max_features))]),
         ["Titulo", "Noticia"]),
        ("autor", Pipeline([("normalizar", FunctionTransformer(prepare_authors, validate=False)),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), ["Autor"])
    ], sparse_threshold=1.0)
    return Pipeline([("schema", InputSchemaValidator()), ("columns", columns)])


REVIEW_POLICY = "v1_texto_autor_pre_checagem_1"
CHECKING_PATTERN = re.compile(
    r"\b(?:boatos?|fals[oa]s?|fake(?:\s+news)?|enganos[oa]s?|checagem|"
    r"desmentid[oa]s?|montagem|e[ -]?farsas|comprova)\b", re.IGNORECASE)

def review_template(frame):
    sheet = frame[TRAIN_COLUMNS].copy()
    sheet["id_original"] = frame.index
    sheet["hash_original"] = sheet[TRAIN_COLUMNS].fillna("").apply(
        lambda row: hashlib.sha256(json.dumps(row.tolist(), ensure_ascii=False,
                                              default=str).encode("utf-8")).hexdigest(), axis=1)
    sheet["pistas_texto"] = combine_text(sheet).map(
        lambda text: " | ".join(sorted(set(m.group(0).casefold() for m in CHECKING_PATTERN.finditer(text)))))
    for column in ["decisao", "titulo_aprovado", "noticia_aprovada", "autor_original_aprovado",
                   "justificativa", "revisor", "proveniencia_alegacao"]:
        sheet[column] = ""
    return sheet

def reviewed_dataset(frame, root):
    review_path = root / "data/audit/v1_revisao_texto_autor.csv"
    template = review_template(frame)
    if not review_path.exists():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("x", encoding="utf-8", newline="") as output:
            template.to_csv(output, index=False)
        raise RuntimeError(f"Ficha criada em {review_path}. Revise antes do treino; consulte o notebook 04.")
    review = pd.read_csv(review_path, keep_default_na=False)
    columns = ["id_original", "hash_original", "Classe", "decisao", "titulo_aprovado",
               "noticia_aprovada", "autor_original_aprovado", "justificativa", "revisor",
               "proveniencia_alegacao"]
    validate_schema(review, columns, context="ficha de revisão")
    if review.id_original.isna().any() or review.id_original.duplicated().any():
        raise ValueError("IDs ausentes ou duplicados na revisão.")
    expected = template.set_index("id_original")
    review = review.set_index("id_original")
    if set(review.index) != set(expected.index):
        raise ValueError("A ficha deve cobrir todos os registros elegíveis da base original.")
    review = review.loc[expected.index].copy()
    if not review.hash_original.eq(expected.hash_original).all():
        raise ValueError("Base alterada: revisão não corresponde ao texto/autor original.")
    labels = pd.to_numeric(review.Classe, errors="raise")
    if not labels.eq(expected.Classe).all():
        raise ValueError("A revisão não pode modificar os rótulos originais.")
    review["decisao"] = review.decisao.astype(str).str.strip().str.casefold()
    if not review.decisao.isin(["aprovar", "excluir"]).all():
        raise ValueError("Revisão pendente: preencha aprovar/excluir em todas as linhas.")
    for column in ["justificativa", "revisor"]:
        if review[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Campo obrigatório na revisão: {column}.")
    approved = review.loc[review.decisao.eq("aprovar")].copy()
    for column in ["noticia_aprovada", "proveniencia_alegacao"]:
        if approved[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Toda aprovação exige {column}.")
    approved["Titulo"] = approved.titulo_aprovado
    approved["Noticia"] = approved.noticia_aprovada
    # Nunca reutiliza automaticamente o autor da checagem.
    approved["Autor"] = approved.autor_original_aprovado
    approved["Classe"] = labels.loc[approved.index].astype(int)
    if approved.empty or set(approved.Classe) != {0, 1}:
        raise ValueError("A revisão precisa preservar exemplos das duas classes.")
    approved["pistas_residuais"] = combine_text(approved).map(
        lambda text: bool(CHECKING_PATTERN.search(text)))
    print("Revisão:", {"aprovados": len(approved), "excluídos": len(review) - len(approved),
                       "aprovados_com_pistas_residuais": int(approved.pistas_residuais.sum())})
    approved.attrs["review_policy"] = REVIEW_POLICY
    approved.attrs["review_sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    return approved

def _load_original_dataset(root):
    frame = pd.read_excel(root / "data/FakeRecogna.xlsx")
    validate_schema(frame, TRAIN_COLUMNS, context="dataset de treinamento")
    frame = frame.dropna(subset=["Noticia", "Classe"]).copy()
    labels = pd.to_numeric(frame["Classe"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Classe deve ser 0 = Fake ou 1 = Real.")
    frame["Classe"] = labels.astype(int)
    return frame


def _deduplicate_dataset(frame):
    keys = combine_text(frame).map(normalize_text)
    frame = frame.loc[keys.ne("") & frame.Noticia.astype(str).str.strip().ne("")].copy()
    frame["chave_texto"] = combine_text(frame).map(normalize_text)
    if frame.groupby("chave_texto").Classe.nunique().gt(1).any():
        raise ValueError("Textos iguais com rótulos conflitantes exigem revisão.")
    return frame.drop_duplicates("chave_texto")


def load_exploratory_dataset(root):
    """Carrega a base original sem ler, criar ou validar a ficha manual."""
    frame = _load_original_dataset(root)
    warnings.warn(
        "EXPERIMENTO EXPLORATÓRIO: base original não auditada integralmente. "
        "Textos e autores podem conter pistas da checagem. "
        "As métricas não comprovam generalização externa.", UserWarning, stacklevel=2)
    frame.attrs["review_mode"] = "exploratory"
    frame.attrs["review_policy"] = "base_original_sem_aprovacao_integral"
    frame.attrs["raw_sha256"] = hashlib.sha256((root / "data/FakeRecogna.xlsx").read_bytes()).hexdigest()
    return _deduplicate_dataset(frame)


def load_dataset(root, review_mode="exploratory"):
    if review_mode == "exploratory":
        return load_exploratory_dataset(root)
    if review_mode != "strict":
        raise ValueError("review_mode deve ser strict ou exploratory.")
    frame = reviewed_dataset(_load_original_dataset(root), root)
    frame.attrs["review_mode"] = "strict"
    return _deduplicate_dataset(frame)
