import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Motor de explicabilidade para modelos lineares baseados em TF-IDF.
    Gera as explicações no formato exato solicitado pelo front-end.
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
            self.pipeline = self.svc_model.calibrated_classifiers_[0].estimator
            self.vectorizer = self.pipeline.named_steps["tfidf"]
            self.classifier = self.pipeline.named_steps["clf"]
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.coefs = self.classifier.coef_[0]
            logger.info("XAI Engine inicializado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao inicializar XAI: {e}")

    def explain(self, text: str, url: str, prob_fake: float, bias_label: str, top_k: int = 4) -> dict:
        """
        Gera as estruturas janela_analise, o_que_sustenta, o_que_enfraquece, etc.
        """
        # Fallback dictionary
        result = {
            "janela_analise": {
                "evidencia": "Indisponível (Erro na extração).",
                "qualidade_da_fonte": "Não avaliada.",
                "corroboracao": "Não avaliada.",
                "contexto": "Indisponível.",
                "atualidade": "Não avaliada."
            },
            "o_que_sustenta": [],
            "o_que_enfraquece": [],
            "o_que_nao_foi_confirmado": ["A veracidade do texto não pôde ser confirmada automaticamente."]
        }
        
        if self.vectorizer is None or self.classifier is None:
            return result

        try:
            # 1. Obter domínio
            domain = urlparse(url).netloc if url else "domínio desconhecido"
            if "globo.com" in domain or "gov.br" in domain or "uol.com.br" in domain or "estadao" in domain or "folha" in domain:
                qualidade_fonte = f"O domínio {domain} é tradicionalmente reconhecido como veículo de imprensa ou fonte oficial."
            else:
                qualidade_fonte = f"O domínio {domain} requer verificação adicional, pois não está na lista principal de grande mídia ou órgãos oficiais."
                
            # 2. Corroboração via Viés
            if bias_label == "BIASED":
                corroboracao = "O modelo semântico detectou forte presença de viés linguístico, o que afeta a neutralidade da informação."
            elif bias_label == "NEUTRAL":
                corroboracao = "O modelo não detectou viés linguístico significativo. A linguagem parece neutra."
            else:
                corroboracao = "Avaliação de viés não conclusiva."
                
            # 3. Contexto (tamanho do texto)
            palavras_count = len(text.split())
            contexto = f"A notícia analisada possui {palavras_count} palavras processadas."
            
            # 4. TF-IDF
            X = self.vectorizer.transform([text])
            non_zero_indices = X.nonzero()[1]
            
            contributions = []
            for idx in non_zero_indices:
                word = self.feature_names[idx]
                contrib = X[0, idx] * self.coefs[idx]
                contributions.append((word, contrib))
                
            contributions.sort(key=lambda x: x[1])
            top_fake_words = [word for word, contrib in contributions[:top_k] if contrib < 0]
            top_real_words = [word for word, contrib in contributions[-top_k:] if contrib > 0]
            top_real_words = list(reversed(top_real_words))

            # Lógica para o que sustenta e enfraquece baseada na probabilidade
            if prob_fake >= 0.5:
                # O modelo diz que é FAKE.
                # Sustenta = Palavras que empurraram para Fake.
                # Enfraquece = Palavras que empurraram para Real (Factual)
                sustenta = [f"Uso de termos suspeitos ou urgentes: {w}" for w in top_fake_words]
                enfraquece = [f"Presença de termos comuns: {w}" for w in top_real_words]
                evidencia_texto = "O algoritmo identificou forte correlação matemática com padrões de desinformação." if top_fake_words else "Alta probabilidade baseada na estrutura textual como um todo."
            else:
                # O modelo diz que é REAL.
                # Sustenta = Palavras que empurraram para Real.
                # Enfraquece = Palavras que empurraram para Fake (se houver)
                sustenta = [f"Linguagem consistente e descritiva: {w}" for w in top_real_words]
                enfraquece = [f"Termo que gerou leve alerta: {w}" for w in top_fake_words]
                evidencia_texto = "A análise lexical sugere forte alinhamento com reportagens factuais e descritivas." if top_real_words else "Probabilidade baixa de manipulação."

            result["janela_analise"]["evidencia"] = evidencia_texto
            result["janela_analise"]["qualidade_da_fonte"] = qualidade_fonte
            result["janela_analise"]["corroboracao"] = corroboracao
            result["janela_analise"]["contexto"] = contexto
            result["janela_analise"]["atualidade"] = "A ferramenta analisa apenas a escrita, portanto a atualidade dos fatos requer pesquisa manual externa."
            
            result["o_que_sustenta"] = sustenta if sustenta else ["Estrutura matemática geral do texto."]
            result["o_que_enfraquece"] = enfraquece if enfraquece else ["Nenhum fator redutor significativo encontrado."]
            result["o_que_nao_foi_confirmado"] = [
                "Nomes próprios, dados estatísticos e datas presentes no texto não foram verificados contra o mundo real (sem uso de busca web)."
            ]
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na extração matemática de features XAI: {e}")
            return result
