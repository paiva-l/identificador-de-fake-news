import logging

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Motor de explicabilidade para modelos lineares baseados em TF-IDF.
    Extrai as palavras que mais contribuíram matematicamente para a classificação,
    sem utilizar LLMs externos.
    """
    def __init__(self, svc_model):
        self.svc_model = svc_model
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        self.feature_names = None
        self.coefs = None
        
        self._initialize()

    def _initialize(self):
        if self.svc_model is None:
            return
            
        try:
            # O modelo base está empacotado em um CalibratedClassifierCV(cv="prefit")
            # Extraímos o pipeline do primeiro e único calibrador treinado
            self.pipeline = self.svc_model.calibrated_classifiers_[0].estimator
            self.vectorizer = self.pipeline.named_steps["tfidf"]
            self.classifier = self.pipeline.named_steps["clf"]
            self.feature_names = self.vectorizer.get_feature_names_out()
            # Os coeficientes do LinearSVC
            self.coefs = self.classifier.coef_[0]
            logger.info("XAI Engine inicializado com sucesso (LinearSVC Coefs).")
        except Exception as e:
            logger.error(f"Erro ao inicializar XAI: {e}")

    def explain(self, text: str, prob_fake: float, top_k: int = 4) -> str:
        """
        Calcula a contribuição local das features (palavras) no texto de entrada
        e retorna um comentário fixo preenchido com as palavras-chave.
        """
        if self.vectorizer is None or self.classifier is None:
            return "Explicabilidade indisponível (modelo base não carregado ou não é linear)."

        try:
            # 1. Transformar texto no vetor TF-IDF exato
            X = self.vectorizer.transform([text])
            
            # 2. Obter índices de palavras presentes na notícia
            non_zero_indices = X.nonzero()[1]
            
            if len(non_zero_indices) == 0:
                return self._generate_template(prob_fake, [])
                
            # 3. Calcular a contribuição: (Valor TF-IDF) * (Peso da Classe no Modelo)
            contributions = []
            for idx in non_zero_indices:
                word = self.feature_names[idx]
                contrib = X[0, idx] * self.coefs[idx]
                contributions.append((word, contrib))
                
            # 4. Ordenação
            # O Scikit-Learn assume '0' (Fake) e '1' (Verdadeira).
            # Coeficientes negativos empurram o output para 0. Coeficientes positivos para 1.
            contributions.sort(key=lambda x: x[1])
            
            # Top palavras falsas (valores mais negativos)
            top_fake_words = [word for word, contrib in contributions[:top_k] if contrib < 0]
            
            if prob_fake < 0.35:
                # Top palavras verdadeiras (valores mais positivos)
                top_real_words = [word for word, contrib in contributions[-top_k:] if contrib > 0]
                top_words = list(reversed(top_real_words))
            else:
                top_words = top_fake_words
                
            return self._generate_template(prob_fake, top_words)
            
        except Exception as e:
            logger.error(f"Erro na extração matemática de features XAI: {e}")
            return "Erro ao gerar explicação estruturada."

    def _generate_template(self, prob_fake: float, words: list) -> str:
        if not words:
            if prob_fake >= 0.70:
                return "Alta probabilidade de desinformação baseada na estrutura do texto, embora nenhum termo específico tenha se destacado de forma isolada."
            elif prob_fake < 0.35:
                return "A notícia possui linguagem segura e descritiva. Provável notícia real."
            else:
                return "Atenção (Sinal Amarelo). Verifique a fonte oficial da notícia."

        words_str = ", ".join([f"'{w.upper()}'" for w in words])
        
        if prob_fake >= 0.70:
            return f"Nossos algoritmos matemáticos identificaram alta possibilidade de desinformação. Os termos que mais influenciaram o alerta de manipulação foram: {words_str}."
        elif prob_fake >= 0.35:
            return f"O texto apresentou características mistas. Embora não seja definitivamente falso, detectou-se o uso de termos comuns em desinformação como: {words_str}. Recomendamos cautela."
        else:
            return f"A linguagem analisada é predominantemente segura e factual. O modelo destacou o uso de termos neutros/jornalísticos como: {words_str}. Avaliado como provável notícia real."
