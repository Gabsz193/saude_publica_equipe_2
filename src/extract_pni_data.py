"""
Script para extração de dados da API de Dados Abertos do Ministério da Saúde
Endpoint: Vacinação - Doses Aplicadas PNI
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import pandas as pd
from datetime import datetime

API_BASE_URL = "https://apidadosabertos.saude.gov.br/vacinacao"
TARGET_YEARS = [2023, 2024, 2025, 2026]
RECORDS_PER_YEAR = 4000  # 4 requisições de 1000 por ano = 16.000 registros
BATCH_SIZE = 1000

SELECTED_FIELDS = [
    "data_vacina",
    "data_entrada_rnds",
    "sigla_uf_paciente",
    "codigo_municipio_paciente",
    "sigla_uf_estabelecimento",
    "codigo_municipio_estabelecimento",
    "descricao_tipo_estabelecimento",
    "descricao_natureza_estabelecimento",
    "descricao_sistema_origem",
    "numero_idade_paciente",
    "tipo_sexo_paciente",
    "nome_raca_cor_paciente",
    "descricao_vacina_fabricante",
    "codigo_vacina",
    "codigo_dose_vacina",
    "codigo_estrategia_vacinacao",
    "descricao_condicao_maternal"
]

def fetch_page(year: int, offset: int, limit: int = BATCH_SIZE, max_retries: int = 3):
    url = f"{API_BASE_URL}/doses-aplicadas-pni-{year}?limit={limit}&offset={offset}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FIA-PublicHealth/1.0",
        "Accept": "application/json"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                records = data.get("doses_aplicadas_pni", [])
                return records
        except Exception as e:
            print(f"  [Aviso] Tentativa {attempt}/{max_retries} falhou para ano {year}, offset {offset}: {e}")
            time.sleep(2 * attempt)
            
    print(f"  [Erro] Falha definitiva para ano {year}, offset {offset}")
    return []

def extract_all():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    all_rows = []
    print("=== Iniciando Extração de Dados PNI (Ministério da Saúde) ===")
    
    for year in TARGET_YEARS:
        print(f"\n--- Coletando dados do ano: {year} ---")
        pages = RECORDS_PER_YEAR // BATCH_SIZE
        year_records = []
        
        for p in range(pages):
            offset = p
            print(f"  Buscando página {p+1}/{pages} (offset={offset}, limit={BATCH_SIZE})...", end="", flush=True)
            t0 = time.time()
            records = fetch_page(year, offset=offset, limit=BATCH_SIZE)
            elapsed = time.time() - t0
            print(f" {len(records)} registros retornados em {elapsed:.1f}s")
            
            for r in records:
                # Filtrar apenas os campos selecionados e adicionar ano
                row = {k: r.get(k) for k in SELECTED_FIELDS}
                row["ano_coleta"] = year
                year_records.append(row)
                
            time.sleep(0.5)  # Respeito à taxa da API
            
        print(f"Total obtido para {year}: {len(year_records)} registros")
        all_rows.extend(year_records)
        
    df = pd.DataFrame(all_rows)
    print(f"\nTotal consolidado: {len(df)} registros e {df.shape[1]} colunas.")
    
    parquet_path = os.path.join(output_dir, "vacinacao_pni_amostra.parquet")
    csv_path = os.path.join(output_dir, "vacinacao_pni_amostra.csv.gz")
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, compression="gzip")
    
    print(f"Arquivo Parquet salvo em: {parquet_path}")
    print(f"Arquivo CSV compactado salvo em: {csv_path}")
    print("=== Extração Concluída com Sucesso! ===")
    return df

if __name__ == "__main__":
    extract_all()
