import pandas as pd
import sqlite3
import logging
from ml.train import run_training
from ml.preprocessing import limpar_texto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_continuous_training():
    logger.info("Verificando banco de dados em busca de novos feedbacks da comunidade...")
    
    # Conecta no SQLite de Produção (para o cenário atual)
    # Numa aplicação real, conectaríamos ao PostgreSQL via variável de ambiente
    conn = sqlite3.connect("fake_news_local.db")
    
    # Busca notícias que receberam votos discordantes do modelo
    # Supondo que voto = 0 seja Fake e voto = 1 seja Real
    query = """
    SELECT n.conteudo, v.voto
    FROM noticias_cache n
    JOIN votos_comunidade v ON n.id = v.id_noticia
    """
    
    try:
        df_feedbacks = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Erro ao acessar banco de dados: {e}")
        return
    finally:
        conn.close()
        
    if df_feedbacks.empty:
        logger.info("Nenhum feedback novo encontrado. Retreinamento não é necessário no momento.")
        return

    logger.info(f"Foram encontrados {len(df_feedbacks)} novos registros de feedback.")
    
    # Mapear os dados do banco para o padrão do dataset
    # Renomear 'conteudo' para 'texto_limpo' e 'voto' para 'Classe'
    df_feedbacks = df_feedbacks.rename(columns={'conteudo': 'texto_limpo', 'voto': 'Classe'})
    
    # Aplicar a mesma limpeza rigorosa
    df_feedbacks['texto_limpo'] = df_feedbacks['texto_limpo'].apply(lambda x: limpar_texto(str(x)))
    
    # Carregar dataset original versionado
    dataset_path = "data/fake_recogna_limpo.csv"
    try:
        df_original = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.error("Dataset original não encontrado. Rode 'dvc pull' primeiro.")
        return
        
    # Anexar os novos dados
    logger.info("Mesclando novos registros ao dataset oficial...")
    df_atualizado = pd.concat([df_original, df_feedbacks], ignore_index=True)
    df_atualizado = df_atualizado.drop_duplicates(subset=['texto_limpo'], keep='last')
    
    # Salvar o dataset atualizado
    df_atualizado.to_csv(dataset_path, index=False)
    logger.info("Dataset atualizado salvo. Iniciando o Pipeline de MLflow...")
    
    # Acionar o pipeline do MLflow para treinar a nova versão do modelo
    run_training()
    
    logger.info("Retreinamento Contínuo finalizado com sucesso!")
    # Observação: Aqui, poderíamos automatizar um 'dvc add' e 'git commit' 
    # ou deixar para aprovação humana.

if __name__ == "__main__":
    execute_continuous_training()
