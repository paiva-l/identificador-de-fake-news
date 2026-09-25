import streamlit as st
import joblib
import os
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Identificador de Fake News",
    page_icon="🛡️",
    layout="centered"
)

# Função para carregar o modelo treinado com cache do Streamlit
@st.cache_resource
def load_model():
    # Lista de caminhos possíveis para encontrar o modelo serializado
    model_paths = [
        "models/modelo_linearsvc_fakerecogna.joblib",
        "modelo_linearsvc_fakerecogna.joblib",
        "../models/modelo_linearsvc_fakerecogna.joblib"
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# Cabeçalho da aplicação
st.title("🛡️ Identificador Inteligente de Fake News")
st.markdown("""
Cole o texto de uma notícia abaixo para analisar a probabilidade de ser **Verdadeira** ou **Falsa**, 
utilizando o pipeline de Machine Learning (`LinearSVC` otimizado via GridSearch).
""")

# Caixa de texto para entrada do usuário
texto_usuario = st.text_area(
    "Digite ou cole o texto da notícia aqui:", 
    height=180, 
    placeholder="Ex: Cientistas descobrem nova tecnologia quântica revolucionária em laboratório de Brasília..."
)

# Botão de execução
if st.button("Analisar Notícia", type="primary"):
    if not texto_usuario.strip():
        st.warning("⚠️ Por favor, insira um texto válido para análise.")
    elif model is None:
        st.error("❌ **Modelo não encontrado!** Certifique-se de que o arquivo `.joblib` está salvo na pasta `models/`.")
    else:
        with st.spinner("Analisando padrões textuais e margens de decisão..."):
            # Predição da classe (0 para Real, 1 para Fake - ajuste conforme seu label encoding)
            predicao = model.predict([texto_usuario])[0]
            
            # Extração da distância ao hiperplano (.decision_function) para estimar a confiança
            try:
                margem = model.decision_function([texto_usuario])[0]
                # Conversão sigmoid simples para estimativa de certeza visual (0 a 100%)
                confianca = 1 / (1 + np.exp(-abs(margem))) * 100
            except Exception:
                margem = 0.0
                confianca = 50.0

        st.markdown("---")
        st.subheader("📊 Resultado da Auditoria")

        # Exibição visual do resultado
        if predicao == 1:
            st.error("🚨 **ALERTA: Possível Fake News Detectada!**")
            st.markdown("O modelo identificou padrões textuais e termos estatisticamente associados a notícias falsas.")
        else:
            st.success("✅ **Notícia Provavelmente Verdadeira**")
            st.markdown("O modelo identificou padrões consistentes com o vocabulário de textos jornalísticos autênticos.")

        # Métricas de suporte ao usuário
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Distância ao Hiperplano", value=f"{margem:.4f}")
        with col2:
            st.metric(label="Certeza Estimada do Modelo", value=f"{confianca:.1f}%")
        
        st.info("💡 **Nota Ética:** Esta ferramenta tem fins acadêmicos e de pesquisa. O resultado reflete probabilidades estatísticas e deve ser validado criticamente.")

# Rodapé profissional
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Desenvolvido com Python, Scikit-Learn e Streamlit 🚀</p>", unsafe_allow_html=True)