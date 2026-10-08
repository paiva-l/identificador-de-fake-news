import os
import logging
import joblib
import numpy as np
from transformers import pipeline
from ml.xai import ExplainabilityEngine
from ml.preprocessing import limpar_texto

logger = logging.getLogger(__name__)

class NLPEngine:
    def __init__(self):
        # Modelo 1: HuggingFace (Análise de Viés e Nuances)
        self.hf_classifier = None
        self.hf_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        
        # Modelo 2: LinearSVC (Detecção FakeNews Factual Treinado no BR)
        self.svc_model = None
        self.tfidf_vectorizer = None
        
        # Motor XAI
        self.xai_engine = None

    def load_model(self):
        """
        Carrega os modelos na memória. Deve ser chamado no lifespan do FastAPI.
        """
        logger.info(f"Carregando pesos do modelo HuggingFace ({self.hf_model_name})...")
        self.hf_classifier = pipeline(
            "zero-shot-classification",
            model=self.hf_model_name
        )
        
        logger.info("Carregando LinearSVC + TF-IDF Calibrado (Dataset BR)...")
        # Caminho relativo baseado na raiz do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        svc_path = os.path.join(base_dir, "models", "calibrated_pipeline.joblib")
        
        try:
            self.svc_model = joblib.load(svc_path)
            self.xai_engine = ExplainabilityEngine(self.svc_model)
            logger.info("LinearSVC Calibrado carregado com sucesso.")
        except Exception as e:
            logger.error(f"Aviso: Não foi possível carregar o LinearSVC. A IA principal assume tudo. Erro: {e}")

        logger.info("Motor Híbrido NLP pronto para inferência.")

    def predict(self, text: str, url: str = "") -> dict:
        """
        Realiza a inferência usando Ensemble.
        LinearSVC: Calcula probabilidade matemática de ser Falso.
        HuggingFace: Extrai a característica de viés / sensacionalismo.
        """
        if not self.hf_classifier:
            raise RuntimeError("O motor Híbrido não foi inicializado. Chame load_model() primeiro.")

        # Limpeza avançada (usada no treino) exclusivamente para o modelo linear
        texto_limpo = limpar_texto(text)

        # --- AVALIAÇÃO 1: VIÉS INFORMATIVO (Hugging Face) ---
        text_cut = text[:1500] # Limite de RAM da GPU/CPU pro BERT
        bias_labels = ["neutro, imparcial, informativo", "opinativo, enviesado, sensacionalista"]
        bias_result = self.hf_classifier(text_cut, bias_labels, multi_label=False)
        
        bias_label_short = "NEUTRAL" if "neutro" in bias_result["labels"][0] else "BIASED"

        # --- AVALIAÇÃO 2: FAKE NEWS (LinearSVC Calibrado) ---
        prob_fake = None
        explicacao = None
        if self.svc_model is not None:
            try:
                # O CalibratedClassifierCV suporta predict_proba. classes_ = [0, 1] onde 0 é Fake e 1 é Real
                probs = self.svc_model.predict_proba([texto_limpo])[0]
                prob_fake = float(probs[0])
                
                # Chamando o motor XAI para gerar o dicionário exato para o Front-end
                if self.xai_engine:
                    explicacao = self.xai_engine.explain(texto_limpo, url, prob_fake, bias_label_short)
            except Exception as e:
                logger.error(f"Erro na inferência do SVC: {e}")
            
        # Fallback caso o modelo tradicional não tenha sido encontrado
        if prob_fake is None:
            rel_labels = ["notícia verdadeira, fato", "mentira, falso, boato"]
            rel_result = self.hf_classifier(text_cut, rel_labels, multi_label=False)
            prob_fake = rel_result["scores"][0] if rel_result["labels"][0] == "mentira, falso, boato" else 1.0 - rel_result["scores"][0]

        return {
            "prob_fake": round(prob_fake, 4),
            "bias_label": bias_result["labels"][0].split(',')[0],
            "bias_score": round(bias_result["scores"][0], 4),
            "explicacao_xai": explicacao,
            "modelos_usados": ["LinearSVC (BR)", "mDeBERTa-v3 (Zero-Shot)"]
        }

# Instância Singleton do motor
engine = NLPEngine()
