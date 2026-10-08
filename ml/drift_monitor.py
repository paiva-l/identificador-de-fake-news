import pandas as pd
import sqlite3
import os
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_drift_report():
    logger.info("Iniciando geração de relatório de Data Drift...")
    
    # 1. Carregar dados de treinamento (Referência)
    ref_path = "data/fake_recogna_limpo.csv"
    if not os.path.exists(ref_path):
        logger.error(f"Arquivo de referência não encontrado: {ref_path}")
        return
        
    df_ref = pd.read_csv(ref_path)
    
    # Padronizar colunas de referência
    col_text_ref = 'texto_limpo' if 'texto_limpo' in df_ref.columns else 'Noticia'
    if col_text_ref not in df_ref.columns:
        col_text_ref = df_ref.columns[0]
        
    df_ref = df_ref.rename(columns={col_text_ref: 'texto'})
    
    # Usar a Classe como proxy para 'prediction' caso não haja score original
    col_label = 'Classe' if 'Classe' in df_ref.columns else df_ref.columns[-1]
    if col_label in df_ref.columns:
        df_ref['prediction'] = df_ref[col_label]
    else:
        df_ref['prediction'] = 0.0
    
    # Selecionar apenas uma amostra se o dataset for muito grande (otimização)
    if len(df_ref) > 5000:
        df_ref = df_ref.sample(5000, random_state=42)
        
    # 2. Carregar dados em produção (Atual)
    db_path = "fake_news_local.db"
    if not os.path.exists(db_path):
        logger.error(f"Banco de dados de produção não encontrado: {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    # Lendo as notícias mais recentes (que o modelo inferiu em produção)
    df_cur = pd.read_sql_query("SELECT conteudo as texto, score_ml as prediction FROM noticias_cache WHERE conteudo IS NOT NULL", conn)
    conn.close()
    
    if len(df_cur) == 0:
        logger.warning("Não há dados suficientes em produção para calcular o Drift.")
        return
        
    logger.info(f"Dados carregados. Referência: {len(df_ref)} linhas, Produção: {len(df_cur)} linhas.")
    
    # Preencher NaN
    df_ref['texto'] = df_ref['texto'].fillna('')
    df_cur['texto'] = df_cur['texto'].fillna('')
    df_ref['prediction'] = df_ref['prediction'].fillna(0.0)
    df_cur['prediction'] = df_cur['prediction'].fillna(0.0)

    # 3. Configurar Mapeamento de Colunas
    column_mapping = ColumnMapping()
    column_mapping.text_features = ['texto']
    column_mapping.numerical_features = ['prediction']
    column_mapping.categorical_features = []
    # Remover target porque a produção não tem ground truth anotado ainda
    column_mapping.target = None 

    # 4. Gerar Relatório
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # Manter colunas idênticas
    df_ref_sub = df_ref[['texto', 'prediction']]
    df_cur_sub = df_cur[['texto', 'prediction']]
    
    logger.info("Executando cálculos do Evidently AI...")
    report.run(reference_data=df_ref_sub, current_data=df_cur_sub, column_mapping=column_mapping)
    
    # 5. Salvar Dashboard HTML
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/data_drift_report.html"
    report.save_html(report_path)
    
    logger.info(f"Relatório de Data Drift gerado com sucesso em: {report_path}")

if __name__ == "__main__":
    generate_drift_report()
