"""
Funções utilitárias para o trabalho de Fundamentos de IA:
- Carregamento e higienização dos dados do PNI
- Engenharia de atributos
- Métricas e avaliação de modelos
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict

UF_TO_REGION = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

def load_and_preprocess_pni(data_path: str = None) -> pd.DataFrame:
    """Carrega os dados extraídos da API e realiza a limpeza e engenharia de features."""
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        parquet_file = os.path.join(base_dir, "data", "vacinacao_pni_amostra.parquet")
        csv_file = os.path.join(base_dir, "data", "vacinacao_pni_amostra.csv.gz")
        data_path = parquet_file if os.path.exists(parquet_file) else csv_file
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em {data_path}. Execute src/extract_pni_data.py primeiro.")
        
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        
    print(f"[Data Loader] {len(df)} registros brutos carregados.")
    
    # 1. Tratamento de Datas e Cálculo de Latência (Dias de Atraso)
    df['dt_vacina'] = pd.to_datetime(df['data_vacina'], format='mixed', errors='coerce', utc=True)
    df['dt_entrada'] = pd.to_datetime(df['data_entrada_rnds'], format='mixed', errors='coerce', utc=True)
    
    # Descartar registros sem datas válidas
    df = df.dropna(subset=['dt_vacina', 'dt_entrada']).copy()
    
    # Diferença em dias
    df['dias_atraso'] = (df['dt_entrada'] - df['dt_vacina']).dt.total_seconds() / 86400.0
    
    # Filtrar inconsistências cronológicas (atraso negativo) e outliers extremos (> 365 dias)
    df = df[(df['dias_atraso'] >= 0) & (df['dias_atraso'] <= 365)].copy()
    
    # Variável Alvo Binária: Atraso Crítico (> 7 dias = Notificação Tardia)
    df['atraso_critico'] = (df['dias_atraso'] > 7.0).astype(int)
    
    # 2. Engenharia de Features
    # Idade numérica
    idade_num = pd.to_numeric(df['numero_idade_paciente'], errors='coerce')
    med_idade = idade_num.median() if len(idade_num.dropna()) > 0 else 30.0
    df['idade'] = idade_num.fillna(med_idade).clip(0, 105)
    
    # Faixa etária categórica
    bins = [-1, 4, 17, 59, 120]
    labels = ['Crianca', 'Jovem', 'Adulto', 'Idoso']
    df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels).astype(str)
    
    # Região do Estabelecimento
    df['uf'] = df['sigla_uf_estabelecimento'].fillna(df['sigla_uf_paciente']).fillna('SP')
    df['regiao'] = df['uf'].map(UF_TO_REGION).fillna('Outra')
    
    # Tipo de Estabelecimento
    def classificar_estabelecimento(tipo):
        if not isinstance(tipo, str):
            return 'UBS_Centro_Saude'
        tipo_upper = tipo.upper()
        if 'CENTRO DE SAUDE' in tipo_upper or 'UNIDADE BASICA' in tipo_upper or 'POSTO' in tipo_upper:
            return 'UBS_Centro_Saude'
        elif 'HOSPITAL' in tipo_upper or 'PRONTO SOCORRO' in tipo_upper:
            return 'Hospital_Pronto_Socorro'
        elif 'CLINICA' in tipo_upper or 'CONSULTORIO' in tipo_upper:
            return 'Clinica_Ambulatorio'
        return 'Outro'
        
    df['tipo_estabelecimento'] = df['descricao_tipo_estabelecimento'].apply(classificar_estabelecimento)
    
    # Sistema de Origem Offline vs Online
    df['sistema_offline'] = df['descricao_sistema_origem'].astype(str).str.upper().apply(
        lambda s: 1 if 'OFFLINE' in s else 0
    )
    
    # Tipo de Dose (1ª dose, 2ª dose, Reforço, Outra)
    def classificar_dose(dose):
        d_str = str(dose).strip()
        if d_str in ['1', '01']:
            return '1a_Dose'
        elif d_str in ['2', '02']:
            return '2a_Dose'
        elif d_str in ['3', '38', '03', 'reforco', 'REFORCO']:
            return 'Reforco'
        return 'Outra'
        
    df['tipo_dose'] = df['codigo_dose_vacina'].apply(classificar_dose)
    
    # Fabricante / Vacina Agrupada
    def classificar_fabricante(fab):
        if not isinstance(fab, str):
            return 'Outro'
        f = fab.upper()
        if 'PFIZER' in f:
            return 'Pfizer'
        elif 'BUTANTAN' in f:
            return 'Butantan'
        elif 'FIOCRUZ' in f or 'OSWALDO CRUZ' in f:
            return 'Fiocruz'
        elif 'SERUM' in f or 'INDIA' in f:
            return 'Serum_India'
        return 'Outro'
        
    df['fabricante'] = df['descricao_vacina_fabricante'].apply(classificar_fabricante)
    
    # Deslocamento: paciente reside no mesmo município onde se vacinou?
    df['mesmo_municipio'] = (
        (df['codigo_municipio_paciente'].notna()) & 
        (df['codigo_municipio_paciente'] == df['codigo_municipio_estabelecimento'])
    ).astype(int)
    
    # Sexo binário/limpo
    df['sexo'] = df['tipo_sexo_paciente'].fillna('I').apply(lambda s: s if s in ['M', 'F'] else 'Outro')
    
    print(f"[Data Loader] {len(df)} registros válidos após higienização.")
    print(f"[Distribuição Alvo] Atraso Crítico (>7 dias): {df['atraso_critico'].mean()*100:.1f}% positivo.")
    return df

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula acurácia, precisão, revocação e F1-score do zero."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
