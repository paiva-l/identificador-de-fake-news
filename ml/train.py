import pandas as pd
import logging
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# Importar o pré-processador do pipeline de produção
from ml.preprocessing import limpar_texto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    logger.info("Iniciando pipeline de treinamento unificado...")

    # Configurar MLflow para usar sqlite local
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("FakeNews_Detection_Factual")

    # 1. Carregar dados
    data_path = "data/fake_recogna_limpo.csv"
    if not os.path.exists(data_path):
        logger.error(f"Dataset não encontrado em {data_path}. Certifique-se de executar 'dvc pull'.")
        return

    df = pd.read_csv(data_path)
    # Supondo que a classe seja 0 para Fake e 1 para Real
    # e as colunas sejam texto_limpo e Classe
    # Vamos re-limpar os dados usando nosso novo módulo para sanar o skew
    logger.info("Aplicando ml.preprocessing.limpar_texto...")
    # O dataset bruto original (se fosse usado) estaria em Noticia, mas vamos garantir:
    col_text = 'Noticia' if 'Noticia' in df.columns else df.columns[0]
    col_label = 'Classe' if 'Classe' in df.columns else df.columns[-1]

    # No fake_recogna_limpo, as colunas são texto_limpo e Classe
    if 'texto_limpo' in df.columns:
        X_raw = df['texto_limpo'].fillna('')
    else:
        X_raw = df[col_text].fillna('')

    y = df[col_label].values

    # Preprocessamento simétrico exato
    X_clean = X_raw.apply(lambda t: limpar_texto(str(t)))

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run():
        logger.info("Treinando classificador Calibrated LinearSVC...")
        
        # Hyperparameters
        max_features = 50000
        c_param = 10.0
        
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", c_param)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_df=0.85, max_features=max_features, ngram_range=(1, 2))),
            ('clf', LinearSVC(C=c_param, max_iter=2000, random_state=42))
        ])

        # Envolver no calibrador para extrair probabilidades
        calibrated_model = CalibratedClassifierCV(pipeline, cv=5)
        
        # Treinamento
        calibrated_model.fit(X_train, y_train)

        # Avaliação
        logger.info("Avaliando o modelo...")
        y_pred = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("f1_weighted", f1)
        
        logger.info(f"F1 Score: {f1:.4f}")

        # Salvar o modelo no MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            registered_model_name="Factual_LinearSVC_Pipeline"
        )
        
        logger.info("Modelo salvo e registrado com sucesso no MLflow!")

if __name__ == "__main__":
    run_training()
