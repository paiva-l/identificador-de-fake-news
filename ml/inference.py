import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class NLPEngine:
    def __init__(self):
        self.classifier = None
        # Para este MVP rápido e leve, utilizaremos um modelo zero-shot multilingue.
        self.model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    def load_model(self):
        """
        Carrega o modelo na memória. 
        Este método deve ser chamado apenas uma vez, no evento `lifespan` do FastAPI.
        """
        logger.info(f"Carregando pesos do modelo NLP ({self.model_name})...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name
        )
        logger.info("Modelo NLP carregado e pronto para inferência.")

    def predict(self, text: str) -> dict:
        """
        Realiza a inferência em um texto limpo, gerando o score de probabilidade
        e a classificação de viés/opinião.
        """
        if not self.classifier:
            raise RuntimeError("O modelo de ML não foi carregado. Chame load_model() primeiro.")

        # Truncamento de segurança para não estourar a memória RAM no MVP
        text_cut = text[:1500]

        # 1. Determina se é fato jornalístico genuíno ou mentira/boato (Confiança/Fake)
        rel_labels = ["notícia verdadeira, fato", "mentira, falso, boato"]
        rel_result = self.classifier(text_cut, rel_labels, multi_label=False)
        
        prob_fake = 0.0
        if rel_result["labels"][0] == "mentira, falso, boato":
            prob_fake = rel_result["scores"][0]
        else:
            prob_fake = 1.0 - rel_result["scores"][0]

        # 2. Avalia o viés / câmara de eco (imparcial vs sensacionalista)
        bias_labels = ["neutro, imparcial, informativo", "opinativo, enviesado, sensacionalista"]
        bias_result = self.classifier(text_cut, bias_labels, multi_label=False)
        
        return {
            "prob_fake": round(prob_fake, 4),
            "bias_label": bias_result["labels"][0].split(',')[0],  # Pega o termo principal (ex: "neutro")
            "bias_score": round(bias_result["scores"][0], 4),
            "modelo": self.model_name
        }

# Instância Singleton do motor
engine = NLPEngine()
