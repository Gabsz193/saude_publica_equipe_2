# Trabalho Integrado de Fundamentos de IA — Equipe 2
**Do Simbólico à Incerteza: Pipeline Completo de Inteligência Artificial**  
*Domínio: Saúde Pública (Vigilância Epidemiológica & Imunização PNI)*  
*Professor: Edjard Mota*

---

## 1. Visão Geral do Projeto

Este projeto implementa o pipeline completo de aprendizado de máquina e raciocínio sob incerteza exigido na disciplina de **Fundamentos de Inteligência Artificial**, aplicando-o ao domínio oficial da **Saúde Pública**.

### Pergunta Prática do Domínio
> *"Um registro de vacinação sofrerá atraso crítico de notificação (> 7 dias entre a aplicação e a inserção no sistema nacional RNDS)?"*

A oportunidade da notificação vacinal é um indicador epidemiológico crítico para o SUS: atrasos prolongados geram cegueira epidemiológica para a vigilância sanitária, dificultando o controle de surtos e o cálculo da cobertura vacinal em tempo real.

---

## 2. Estrutura do Pipeline em 6 Etapas

| Etapa | Tema | Algoritmo / Conteúdo Principal | Notebook |
|---|---|---|---|
| **1** | **Aprendizado Simbólico** | Árvore de Decisão do zero (`LEARN-DECISION-TREE`), cálculo manual de Entropia e Ganho de Informação, 5-fold CV (~76.5% acurácia) e análise de não-monotonicidade. | [`etapa1/etapa1.ipynb`](etapa1/etapa1.ipynb) |
| **2** | **Aprendizado Linear** | Regressão Linear Univariada (fórmula fechada OLS), Multivariada com Gradiente Descendente do zero, Regularização L2 (Ridge com trajetória de coeficientes) e Regressão Logística do zero (~77% acurácia). | [`etapa2/etapa2.ipynb`](etapa2/etapa2.ipynb) |
| **3** | **Probabilidade** | Construção da Distribuição Conjunta (4 eventos atômicos, soma=1.0), Marginais, Condicionais, Regra da Cadeia para 3 variáveis, Teorema de Bayes (raciocínio causal vs. diagnóstico) e independência condicional. | [`etapa3/etapa3.ipynb`](etapa3/etapa3.ipynb) |
| **4** | **Aprendizado Probabilístico & EM** | MLE para Bernoulli, MAP com prior Beta(2,2) demonstrando regularização bayesiana em pequenas amostras, Naive Bayes do zero (~75.8% acurácia) e algoritmo Expectation-Maximization (EM) inferindo variável latente operacional $Z$ (~93.8% acurácia com EM). | [`etapa4/etapa4.ipynb`](etapa4/etapa4.ipynb) |
| **5** | **Redes Bayesianas** | DAG causal com 5 nós, estimação de CPTs empíricas, Inferência por Enumeração exata do zero (validada com `pgmpy`), análise de d-separação (demonstração de colisor) e cálculo de economia de parâmetros (10 vs 31 parâmetros, redução de 67.7%). | [`etapa5/etapa5.ipynb`](etapa5/etapa5.ipynb) |
| **6** | **Lógica Fuzzy & Integração** | Sistema de Inferência Fuzzy Mamdani (`scikit-fuzzy`) combinando o Risco Bayesiano da Etapa 5 e a Vulnerabilidade Populacional em um Escore Suave de Urgência de Intervenção, comparação com threshold booleano rígido e superfície 3D de controle. | [`etapa6/etapa6.ipynb`](etapa6/etapa6.ipynb) |

---

## 3. Estrutura de Diretórios

```text
trabalho_fia_final/
├── setup.sh                         # Script automatizado de criação do ambiente virtual e instalação
├── README.md                        # Guia de reprodução e documentação do projeto
├── 2o-Trab-SimbML-Incerteza.pdf     # Especificação oficial do professor
├── 2o-Trab-SimbML-Incerteza.md      # Versão da especificação em Markdown otimizada
│
├── data/                            # Microdados extraídos da API
│   ├── vacinacao_pni_amostra.parquet # Amostra tratada oficial (16.000 registros, 2023 a 2026)
│   └── vacinacao_pni_amostra.csv.gz  # Backup compactado em CSV
│
├── src/                             # Módulos Python reutilizáveis (implementações do zero)
│   ├── extract_pni_data.py          # Script de extração da API de Dados Abertos do Ministério da Saúde
│   ├── utils.py                     # Carga, higienização, engenharia de atributos e métricas
│   ├── decision_tree.py             # Árvore de Decisão ID3 do zero com entropia e ganho de informação
│   ├── linear_models.py             # OLS, Gradiente Descendente com Ridge (L2) e Regressão Logística
│   ├── em.py                        # MLE, MAP, Naive Bayes do zero e algoritmo EM (Mistura Gaussiana 1D)
│   ├── bayesian_net.py              # Rede Bayesiana e algoritmo de Inferência Exata por Enumeração
│   └── fuzzy_system.py              # Sistema Mamdani (pertinências, regras e defuzzificação)
│
├── etapa1/etapa1.ipynb               # Notebook executado da Etapa 1
├── etapa2/etapa2.ipynb               # Notebook executado da Etapa 2
├── etapa3/etapa3.ipynb               # Notebook executado da Etapa 3
├── etapa4/etapa4.ipynb               # Notebook executado da Etapa 4
├── etapa5/etapa5.ipynb               # Notebook executado da Etapa 5
└── etapa6/etapa6.ipynb               # Notebook executado da Etapa 6
```

---

## 4. Instruções de Reprodução

### 4.1 Pré-requisitos
- Python 3.10+ (testado e validado em Python 3.13)
- Conexão com a internet para baixar pacotes e/ou atualizar dados da API.

### 4.2 Instalação Automatizada
Para criar o ambiente virtual `.venv`, atualizar as dependências e registrar o kernel do Jupyter, execute:
```bash
chmod +x setup.sh
./setup.sh
```

### 4.3 (Opcional) Re-extração dos Dados da API
O repositório já inclui a amostra representativa tratada em `data/vacinacao_pni_amostra.parquet`. Caso deseje realizar uma nova coleta diretamente da API de Dados Abertos do Ministério da Saúde:
```bash
source .venv/bin/activate
python src/extract_pni_data.py
```

### 4.4 Execução dos Notebooks
Ative o ambiente virtual e execute os notebooks pelo Jupyter Lab / VS Code ou via linha de comando:
```bash
source .venv/bin/activate

# Para executar qualquer notebook no terminal:
jupyter nbconvert --to notebook --execute --inplace etapa1/etapa1.ipynb
jupyter nbconvert --to notebook --execute --inplace etapa2/etapa2.ipynb
jupyter nbconvert --to notebook --execute --inplace etapa3/etapa3.ipynb
jupyter nbconvert --to notebook --execute --inplace etapa4/etapa4.ipynb
jupyter nbconvert --to notebook --execute --inplace etapa5/etapa5.ipynb
jupyter nbconvert --to notebook --execute --inplace etapa6/etapa6.ipynb
```
