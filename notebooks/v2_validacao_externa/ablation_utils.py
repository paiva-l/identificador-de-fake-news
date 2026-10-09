"""Construção das variantes retreinadas do estudo de ablação V2."""
from typing import cast
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pipeline_utils import validate_schema

EXPERIMENTOS = {
    "original": ("Modelo original", ["Titulo", "Noticia", "Autor"]),
    "sem_autoria": ("Sem autoria", ["Titulo", "Noticia"]),
    "somente_noticia": ("Somente notícia", ["Noticia"]),
    "sem_titulo": ("Sem título", ["Noticia", "Autor"]),
}


def somente_noticia(frame: pd.DataFrame) -> pd.Series:
    validate_schema(frame, ["Noticia"], context="ablação textual")
    return frame["Noticia"].fillna("").astype(str).str.strip()


def construir_variante(original: Pipeline, experimento: str) -> Pipeline:
    """Clona sem estado aprendido; preserva parâmetros e exclui transformers."""
    _, campos = EXPERIMENTOS[experimento]
    modelo = cast(Pipeline, clone(original))
    stack = cast(StackingClassifier, modelo.named_steps["clf"])
    bases = stack.get_params(deep=False)["estimators"]
    for _, base in bases:
        base = cast(Pipeline, base)
        features = cast(Pipeline, base.named_steps["features"])
        columns = cast(ColumnTransformer, features.named_steps["columns"])
        transformers = []
        for nome, transformer, selecao in columns.transformers:
            if nome == "autor" and "Autor" not in campos:
                continue
            if nome == "texto" and "Titulo" not in campos:
                transformer = cast(Pipeline, transformer)
                transformer.set_params(combinar=FunctionTransformer(somente_noticia, validate=False))
                selecao = ["Noticia"]
            transformers.append((nome, transformer, selecao))
        columns.set_params(transformers=transformers)
    return modelo
