#!/bin/bash
set -e

echo "Criando ambiente virtual .venv..."
python3 -m venv .venv

echo "Ativando ambiente virtual..."
source .venv/bin/activate

echo "Atualizando pip e instalando dependências..."
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn ipykernel requests scikit-fuzzy pgmpy pyarrow

echo "Registrando kernel do Jupyter..."
python -m ipykernel install --user --name=fia_env --display-name="Python (FIA Equipe 2)"

echo "Setup concluído com sucesso!"
