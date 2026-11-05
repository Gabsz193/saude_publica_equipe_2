# Trabalho Integrado de Fundamentos de IA
**Do Simbólico à Incerteza: Pipeline Completo em 3 Semanas**  
*Prof. Edjard Mota — 15 de outubro de 2026*

---

## 1. Visão Geral e Filosofia
Este trabalho adota uma filosofia auto-contida: cada equipe percorre todo o pipeline de aprendizado visto no curso — do aprendizado simbólico até a lógica fuzzy — aplicando-o a um domínio de dados oficiais diferente.

| # | Domínio | Dados Oficiais | Portal |
|---|---|---|---|
| **P1** | **Saúde Pública** | **Internações SIH/SUS + IBGE** (ou Vacinação / Dados Abertos Saúde) | [dadosabertos.saude.gov.br](https://dadosabertos.saude.gov.br) |
| P2 | Educação | Censo Escolar/INEP + IBGE | inep.gov.br |
| P3 | Segurança Pública | Ocorrências + IBGE | dadosabertos.mj.gov.br |
| P4 | Meio Ambiente | Desmatamento + INMET | dadosabertos.ibama.gov.br |

Cada equipe escolhe um domínio e percorre as 6 etapas descritas. O escopo cabe em 3 semanas.

---

## 2. O Pipeline em 6 Etapas

| Etapa | Tema | Conteúdo Obrigatório |
|---|---|---|
| **1** | **Aprendizado Simbólico** | Árvore de decisão, entropia, ganho de informação, validação cruzada, limitações da lógica |
| **2** | **Aprendizado Linear** | Regressão linear (uni/multivariada), gradiente descendente, regularização L1 ou L2, regressão logística |
| **3** | **Probabilidade** | Axiomas, eventos atômicos, conjunta, condicional, regra da cadeia, Bayes, independência |
| **4** | **Aprendizado Probabilístico** | MLE, MAP, Naive Bayes do zero, EM com variável oculta |
| **5** | **Redes Bayesianas** | DAG, CPT, d-separação, inferência por enumeração |
| **6** | **Lógica Fuzzy** | Conjuntos fuzzy, regras Se-Então, Mamdani, integração com saídas anteriores |

> **Integração:** Ao final das 6 etapas, as saídas da Etapa 5 (probabilidades Bayesianas) alimentam a Etapa 6 (sistema fuzzy) para produzir a decisão final interpretável.

---

## 3. Formato e Entregáveis
- **Repositório GitHub com:**
  - Notebooks por etapa (`etapa1.ipynb`, `etapa2.ipynb`, ..., `etapa6.ipynb`).
  - Relatório LaTeX (6–8 páginas, formato artigo).
  - Scripts auxiliares (`utils.py`, `em.py`, `fuzzy.py`).
  - `README.md` com instruções completas de reprodução.

---

## 4. Detalhamento por Etapa

### Etapa 1 — Aprendizado Simbólico (Árvore de Decisão)
- **Objetivo:** A partir dos microdados oficiais do domínio escolhido, induzir uma árvore de decisão que responda a uma pergunta prática (ex.: risco de complicação, necessidade de reforço/urgência, cobertura vacinal baixa, etc.).
- **Tarefas:**
  1. Definir a variável-alvo binária ou multiclasse.
  2. Selecionar 4–6 atributos interpretáveis.
  3. Calcular entropia e ganho de informação manualmente para os 3 primeiros níveis da árvore (no relatório).
  4. Implementar `LEARN-DECISION-TREE` do zero (sem usar scikit-learn para a árvore principal).
  5. Comparar com `DecisionTreeClassifier` do scikit-learn.
  6. Aplicar validação cruzada 5-fold e reportar acurácia média.
  7. Escrever meia página sobre monotonicidade: mostrar que um exemplo novo pode invalidar conclusões anteriores (não-monotonicidade do aprendizado indutivo vs. lógica dedutiva).
- **Gabarito esperado:** Acurácia treino 85–92%, teste (5-fold) 72–80%, profundidade ótima 3–5 níveis.

---

### Etapa 2 — Aprendizado Linear
- **Objetivo:** Modelar a relação entre uma variável-alvo contínua e atributos. Aplicar regressão linear com gradiente descendente, regularização, e regressão logística.
- **Tarefas:**
  1. Escolher uma variável-alvo contínua (ex.: tempo até aplicação/atraso, taxa de cobertura, número de doses).
  2. Implementar regressão linear univariada (fórmula fechada).
  3. Implementar regressão multivariada com gradiente descendente do zero.
  4. Aplicar regularização L2 (Ridge) e plotar a curva de coeficientes vs. $\lambda$.
  5. Implementar regressão logística (sigmoide + gradiente) para o problema de classificação da Etapa 1.
  6. Comparar com `LinearRegression` e `LogisticRegression` do scikit-learn.
- **Gabarito esperado:** $R^2$ simples 0.55–0.65, multivariada 0.70–0.80, Ridge 0.72–0.82, Regressão logística acurácia 78–85%.

---

### Etapa 3 — Probabilidade
- **Objetivo:** Estabelecer a base probabilística. Foco em cálculo e interpretação rigorosos.
- **Tarefas:**
  1. Escolher 2 variáveis binárias do domínio e construir a distribuição conjunta (4 eventos atômicos).
  2. Calcular probabilidades marginais e condicionais.
  3. Aplicar a regra da cadeia para 3 variáveis.
  4. Aplicar o Teorema de Bayes em um caso real do domínio ($P(\text{condição} \mid \text{evidência})$).
  5. Verificar independência condicional entre 2 atributos dado a classe.
- **Gabarito esperado:** Tabela conjunta somando 1.0; cálculo de Bayes justificando raciocínio causal vs. diagnóstico; discussão sobre por que a LPO (Lógica de Primeira Ordem) não expressa incerteza ("70% de chance").

---

### Etapa 4 — Aprendizado Probabilístico e EM
- **Objetivo:** Aprender parâmetros probabilísticos dos dados e lidar com variáveis ocultas via algoritmo EM.
- **Tarefas:**
  1. Implementar MLE (Máxima Verossimilhança) para o parâmetro $\theta$ de uma Bernoulli.
  2. Implementar MAP com prior Beta (ex.: $\text{Beta}(2,2)$).
  3. Implementar Naive Bayes do zero para a classificação da Etapa 1 (treinar e testar).
  4. Introduzir uma variável oculta $Z$ (2 ou 3 componentes, ex.: perfil de vulnerabilidade não observado) e implementar EM do zero para inferi-la.
  5. Plotar a log-verossimilhança ao longo das iterações (deve crescer monotonicamente e estabilizar em 15–30 iterações).
  6. Comparar: Naive Bayes puro vs. Naive Bayes com EM (Gabarito: puro 72–78%, com EM 80–86%).

---

### Etapa 5 — Redes Bayesianas
- **Objetivo:** Modelar dependências entre 4–6 variáveis do domínio via DAG e CPTs, respondendo a consultas por inferência.
- **Tarefas:**
  1. Desenhar o DAG com 4–6 nós (justificando cada aresta causalmente).
  2. Estimar as CPTs (Tabelas de Probabilidade Condicional) a partir dos dados.
  3. Implementar inferência por enumeração do zero.
  4. Comparar com `pgmpy` (`VariableElimination`).
  5. Aplicar d-separação para verificar 2 independências condicionais (incluindo caso de colisor / v-structure).
  6. Calcular a economia de parâmetros da rede em relação à distribuição conjunta completa ($2^n - 1$).
- **Gabarito esperado:** DAG coerente, inferência por enumeração idêntica à do pgmpy, demonstração de colisor (observar colisor ativa caminho).

---

### Etapa 6 — Lógica Fuzzy
- **Objetivo:** Concluir o pipeline com um sistema Mamdani combinando as saídas anteriores em uma decisão suave e interpretável.
- **Tarefas:**
  1. Definir 2–3 variáveis linguísticas (ex.: risco, urgência, adesão).
  2. Criar funções de pertinência triangulares/trapezoidais para cada variável.
  3. Escrever 3–5 regras Se-Então com operadores E/OU.
  4. Implementar o sistema com `scikit-fuzzy`.
  5. Alimentar o sistema com as probabilidades Bayesianas da Etapa 5 como entrada.
  6. Comparar a decisão fuzzy com a decisão hard (threshold 0.5).
  7. Discutir a interpretabilidade com especialista do domínio.

---

## 5. Rubrica de Avaliação
- Etapa 1: 12%
- Etapa 2: 15%
- Etapa 3: 10%
- Etapa 4: 18%
- Etapa 5: 18%
- Etapa 6: 15%
- Relatório LaTeX + notebook limpo + GitHub: 7%
- Apresentação final (10 min): 5%

## 6. Recomendações Gerais
- `np.random.seed(42)` sempre.
- Validação cruzada (nunca reportar apenas treino).
- Documentar fontes, data de consulta, filtros aplicados.
