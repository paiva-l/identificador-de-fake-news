import os
import logging
import joblib
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

class NLPEngine:
    def __init__(self):
        # Modelo 1: HuggingFace (Análise de Viés e Nuances)
        self.hf_classifier = None
        self.hf_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        
        # Modelo 2: LinearSVC (Detecção FakeNews Factual Treinado no BR)
        self.svc_model = None
        self.tfidf_vectorizer = None

    def load_model(self):
        """
        Carrega os modelos na memória. Deve ser chamado no lifespan do FastAPI.
        """
        logger.info(f"Carregando pesos do modelo HuggingFace ({self.hf_model_name})...")
        self.hf_classifier = pipeline(
            "zero-shot-classification",
            model=self.hf_model_name
        )
        
        logger.info("Carregando LinearSVC + TF-IDF original (Dataset BR)...")
        # Caminho relativo baseado na raiz do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        svc_path = os.path.join(base_dir, "models", "best_linearsvc_model.joblib")
        tfidf_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        
        try:
            self.svc_model = joblib.load(svc_path)
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            logger.info("LinearSVC carregado com sucesso.")
        except Exception as e:
            logger.error(f"Aviso: Não foi possível carregar o LinearSVC. A IA principal assume tudo. Erro: {e}")

        logger.info("Motor Híbrido NLP pronto para inferência.")

    def predict(self, text: str) -> dict:
        """
        Realiza a inferência usando Ensemble.
        LinearSVC: Calcula probabilidade matemática de ser Falso.
        HuggingFace: Extrai a característica de viés / sensacionalismo.
        """
        if not self.hf_classifier:
            raise RuntimeError("O motor Híbrido não foi inicializado. Chame load_model() primeiro.")

        # --- AVALIAÇÃO 1: FAKE NEWS (LinearSVC) ---
        prob_fake = None
        if self.svc_model is not None:
            # O .joblib original já é um Pipeline (TF-IDF + SVC), enviamos texto direto
            try:
                margin = float(self.svc_model.decision_function([text])[0])
                prob_fake = float(1 / (1 + np.exp(-margin)))
            except Exception as e:
                logger.error(f"Erro na inferência do SVC: {e}")
            
        # Fallback caso o modelo tradicional não tenha sido encontrado
        if prob_fake is None:
            text_cut = text[:1500]
            rel_labels = ["notícia verdadeira, fato", "mentira, falso, boato"]
            rel_result = self.hf_classifier(text_cut, rel_labels, multi_label=False)
            prob_fake = rel_result["scores"][0] if rel_result["labels"][0] == "mentira, falso, boato" else 1.0 - rel_result["scores"][0]

        # --- AVALIAÇÃO 2: VIÉS INFORMATIVO (Hugging Face) ---
        text_cut = text[:1500] # Limite de RAM da GPU/CPU pro BERT
        bias_labels = ["neutro, imparcial, informativo", "opinativo, enviesado, sensacionalista"]
        bias_result = self.hf_classifier(text_cut, bias_labels, multi_label=False)
        
        return {
            "prob_fake": round(prob_fake, 4),
            "bias_label": bias_result["labels"][0].split(',')[0],
            "bias_score": round(bias_result["scores"][0], 4),
            "modelos_usados": ["LinearSVC (BR)", "mDeBERTa-v3 (Zero-Shot)"]
        }

# Instância Singleton do motor
engine = NLPEngine()
