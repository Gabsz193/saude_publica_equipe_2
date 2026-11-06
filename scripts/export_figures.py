#!/usr/bin/env python3
"""
scripts/export_figures.py
=========================
Geração e exportação de todas as figuras científicas em alta resolução
(PDF vetorial e PNG 300 DPI) para o artigo da SBC.

Figuras geradas:
1. figures/pipeline_integrado.pdf / .png:
   Diagrama esquemático completo de integração ponta a ponta das 6 etapas
   do pipeline de IA nos microdados do PNI (com destaque ao acoplamento Bayes -> Fuzzy).
2. figures/etapa2_ridge_trajetoria.pdf / .png:
   Trajetória dos coeficientes de regularização Ridge vs log10(lambda) e curva de R2.
3. figures/etapa4_em_convergencia.pdf / .png:
   Convergência monótona da log-verossimilhança no algoritmo EM e densidades da mistura bimodal.
4. figures/etapa5_rede_bayesiana_dag.pdf / .png:
   Visualização do DAG causal de 5 nós destacando o nó colisor A, análise de d-separação e economia de parâmetros.
5. figures/etapa6_fuzzy_superficie.pdf / .png:
   Painel multi-gráfico com funções de pertinência triangulares, comparação com limiar rígido nos 3 cenários do SUS e superfície 3D de controle.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuração de diretórios e paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_and_preprocess_pni
from linear_models import LinearRegressionGD, r2_score_scratch
from em import ExpectationMaximizationGMM
from bayesian_net import BayesianNetworkScratch
from fuzzy_system import HealthFuzzyInferenceSystem

# Configuração de estilo científico para publicação SBC
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROJECT_ROOT / "data" / "vacinacao_pni_amostra.parquet"


def save_figure(fig: plt.Figure, name: str):
    """Salva a figura em formato PDF vetorial e PNG 300 DPI."""
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"
    
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] Exportado: {pdf_path.name} e {png_path.name}")


# ==============================================================================
# FIGURA 1: Diagrama Esquemático do Pipeline Integrado
# ==============================================================================
def generate_figure_1_pipeline():
    print("Gerando Figura 1: Pipeline Integrado...")
    fig = plt.figure(figsize=(15.5, 10.5), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Paleta de cores estruturada
    c_data_bg = '#F0F4F8'
    c_data_border = '#1A365D'
    c_s1_bg = '#F4F7FB'
    c_s1_border = '#2B6CB0'
    c_s2_bg = '#F0FAF7'
    c_s2_border = '#234E52'
    c_s3_bg = '#FAF7FD'
    c_s3_border = '#553C9A'
    c_s4_bg = '#FFFDF7'
    c_s4_border = '#C05621'
    c_s5_bg = '#FFF8F8'
    c_s5_border = '#9B2C2C'
    c_s6_bg = '#F2F8FD'
    c_s6_border = '#2B6CB0'

    def draw_card(x, y, w, h, title, lines, bg_col, border_col):
        # Sombra suave
        shadow = FancyBboxPatch((x + 0.35, y - 0.35), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.0",
                                facecolor='#CBD5E0', edgecolor='none', alpha=0.4)
        ax.add_patch(shadow)

        # Caixa principal
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.35,rounding_size=1.0",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.8)
        ax.add_patch(box)

        # Barra de título integrada no topo
        header_h = 3.2
        header_box = FancyBboxPatch((x, y + h - header_h), w, header_h,
                                    boxstyle="round,pad=0.35,rounding_size=0.7",
                                    facecolor=border_col, edgecolor=border_col)
        ax.add_patch(header_box)
        ax.text(x + w / 2.0, y + h - header_h / 2.0, title,
                fontsize=9.6, fontweight='bold', color='white', ha='center', va='center')

        # Conteúdo interno
        curr_y = y + h - header_h - 2.0
        for line in lines:
            if line.startswith("•"):
                ax.text(x + 1.6, curr_y, line, fontsize=9.0, color='#1A202C', va='top')
                curr_y -= 2.5
            elif line.startswith("  -"):
                ax.text(x + 3.4, curr_y, line, fontsize=8.4, color='#4A5568', va='top')
                curr_y -= 2.2
            else:
                ax.text(x + 1.6, curr_y, line, fontsize=8.6, color='#4A5568', va='top')
                curr_y -= 2.3

    # 1. Box Superior: Dados do PNI
    draw_card(
        x=10, y=85.0, w=80, h=13.0,
        title="MICRODADOS OFICIAIS DE VACINAÇÃO DO PNI (DATASUS / RNDS)",
        lines=[
            "• 15.403 registros válidos extraídos da API nacional (2023–2026), 18 atributos clínicos e logísticos",
            "• Variável-Alvo: Atraso Crítico de Notificação (atraso > 7 dias: 51,59% vs Oportuno <= 7 dias: 48,41%)",
            "• Engenharia de Atributos: sistema_offline, fabricante, faixa_etaria, mesmo_municipio, regiao"
        ],
        bg_col=c_data_bg, border_col=c_data_border
    )

    # 2. Etapa 1: Simbólico (Árvore ID3)
    draw_card(
        x=4, y=57.0, w=43, h=22.8,
        title="ETAPA 1: APRENDIZADO SIMBÓLICO (ÁRVORE ID3)",
        lines=[
            "• Algoritmo LEARN-DECISION-TREE implementado do zero",
            "• Cálculos manuais passo a passo de Entropia e Ganho de Informação:",
            "  - Nó Raiz: H(S) = 0,9993 bits | Vencedor: sistema_offline (IG = 0,2133 bits)",
            "  - Nível 2: Ramo Online -> fabricante (IG = 0,0178) | Ramo Offline -> mesmo_mun (IG = 0,0845)",
            "• Desempenho 5-Fold: Acurácia Média 76,53% ± 0,33% (profundidade máxima 4)",
            "• Teoria de Não-Monotonicidade: Indução empírica refutável sob novos dados",
            "  (contrastando formalmente com a monotonicidade dedutiva da LPO)"
        ],
        bg_col=c_s1_bg, border_col=c_s1_border
    )

    # 3. Etapa 2: Aprendizado Linear
    draw_card(
        x=53, y=57.0, w=43, h=22.8,
        title="ETAPA 2: APRENDIZADO LINEAR & REGULARIZAÇÃO",
        lines=[
            "• Alvo Contínuo com estabilização logarítmica: y = ln(1 + dias_atraso)",
            "• Regressão Linear Univariada OLS (Fórmula Fechada): R² = 0,2781",
            "• Gradiente Descendente Multivariado: R² = 0,3015 (1.500 iterações)",
            "• Regularização Ridge (L2): Encolhimento assintótico de pesos vs log10(lambda)",
            "  (estabilidade paramétrica em lambda ótimo ≈ 10⁻³, suprimindo coeficientes ruidosos)",
            "• Regressão Logística Sigmoide do Zero: Acurácia = 77,47%, F1 = 78,15%"
        ],
        bg_col=c_s2_bg, border_col=c_s2_border
    )

    # 4. Etapa 3: Teoria da Probabilidade
    draw_card(
        x=4, y=30.5, w=43, h=22.2,
        title="ETAPA 3: TEORIA DA PROBABILIDADE & AXIOMAS",
        lines=[
            "• Distribuição Conjunta P(S, A) de 4 eventos atômicos (soma = 1,000000)",
            "• Probabilidades Condicionais: P(Atraso | Offline) = 77,43% vs Online = 24,44%",
            "  - Risco Relativo: RR = 3,17× mais chance de atraso crítico em postos offline",
            "• Teorema de Bayes Causal vs Diagnóstico: P(Offline | Atraso) = 76,91%",
            "• Regra da Cadeia de 3 Variáveis: P(S, V, A) = 0,146660 (erro absoluto: 2,78e-17)",
            "• Independência Condicional empírica comprovada entre Fabricante e Idade"
        ],
        bg_col=c_s3_bg, border_col=c_s3_border
    )

    # 5. Etapa 4: MLE/MAP & Algoritmo EM
    draw_card(
        x=53, y=30.5, w=43, h=22.2,
        title="ETAPA 4: APRENDIZADO PROBABILÍSTICO & ALGORITMO EM",
        lines=[
            "• Estimação Bernoulli: MLE (0,51591) vs MAP Beta(2,2) (0,51590)",
            "  - Regularização bayesiana do MAP comprovada sob pequenas amostras (n <= 100)",
            "• Naive Bayes Categórico Puro com suavização de Laplace: Acurácia = 75,79%",
            "• Variável Oculta Z modelando Regime de Transmissão (Mistura Gaussiana 1D):",
            "  - Z=0 (Ágil): μ0 = 0,657 (0,9 dias) | Z=1 (Represado): μ1 = 3,404 (29,1 dias)",
            "• Algoritmo EM: Convergência estritamente monótona comprovada (30 iterações)",
            "• Salto de Acurácia com Z latente: 75,79% -> 93,77% (+17,98 p.p.!)"
        ],
        bg_col=c_s4_bg, border_col=c_s4_border
    )

    # 6. Etapa 5: Redes Bayesianas
    draw_card(
        x=4, y=4.0, w=40, h=22.2,
        title="ETAPA 5: REDE BAYESIANA CAUSAL",
        lines=[
            "• DAG Causal de 5 Nós: R -> S -> A <- N <- V",
            "• Nó Colisor A: S e N convergem em A (d-separados a priori)",
            "• d-Separação: Condicionar em A ativa dependência S-N",
            "• CPTs empíricas e economia paramétrica de 67,74% (10 vs 31)",
            "• Inferência Exata por Enumeração P(A=1 | evidência):",
            "  - Rural: 74,58% | Urbano: 30,59% | Fronteira: 17,58%"
        ],
        bg_col=c_s5_bg, border_col=c_s5_border
    )

    # 7. Etapa 6: Lógica Fuzzy Mamdani
    draw_card(
        x=56, y=4.0, w=40, h=22.2,
        title="ETAPA 6: LÓGICA FUZZY (MAMDANI)",
        lines=[
            "• Sistema de Inferência Fuzzy Mamdani interpretável",
            "• Antecedentes: Risco Bayesiano P(A=1|e) e Vulnerabilidade",
            "• Consequente: Escore Suave de Urgência SUS [0 a 100]",
            "• 5 Regras Especialistas Se-Então + Defuzzificação CoG",
            "• Transição contínua sem saltos abruptos de limiar rígido (0,5)",
            "• Resposta aos Cenários SUS:",
            "  - Rural: 64,61 / 100 | Urbano: 38,79 | Fronteira: 23,90"
        ],
        bg_col=c_s6_bg, border_col=c_s6_border
    )

    # Conectores e Setas de Fluxo
    def draw_arrow(x1, y1, x2, y2, text="", color='#4A5568', lw=1.8, style='->'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=style, mutation_scale=16,
                                linewidth=lw, color=color, zorder=5)
        ax.add_patch(arrow)
        if text:
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            ax.text(mid_x, mid_y, text, fontsize=8.2, fontweight='bold',
                    color=color, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=color, alpha=0.92))

    # Setas da base de dados para Etapa 1 e 2
    draw_arrow(28, 85.0, 25, 79.8, text="Atributos Discretos", color='#2B6CB0')
    draw_arrow(72, 85.0, 75, 79.8, text="Atributos Contínuos/Dummies", color='#234E52')

    # Setas verticais entre etapas
    draw_arrow(25, 57.0, 25, 52.7, text="Seleção Simbólica", color='#4A5568')
    draw_arrow(75, 57.0, 75, 52.7, text="Parâmetros & Baselines", color='#4A5568')
    draw_arrow(25, 30.5, 24, 26.2, text="Distribuições Conjuntas", color='#4A5568')
    draw_arrow(75, 30.5, 76, 26.2, text="Tratamento de Incerteza", color='#4A5568')

    # ==========================================================================
    # DESTAQUE CENTRAL: Acoplamento Direto Etapa 5 -> Etapa 6
    # ==========================================================================
    coup_box = FancyBboxPatch((44.8, 6.8), 10.4, 15.6,
                              boxstyle="round,pad=0.35,rounding_size=0.8",
                              facecolor='#FFFAF0', edgecolor='#DD6B20', linewidth=2.2)
    ax.add_patch(coup_box)

    # Mini header na caixa de acoplamento
    coup_hdr = FancyBboxPatch((44.8, 19.8), 10.4, 2.6,
                              boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor='#DD6B20', edgecolor='#DD6B20')
    ax.add_patch(coup_hdr)
    ax.text(50.0, 21.1, "ACOPLAMENTO", fontsize=8.2, fontweight='bold',
            color='white', ha='center', va='center')

    ax.text(50.0, 18.2, r"$\mathbf{P(A=1 \mid \mathbf{e})}$", fontsize=11.5, fontweight='bold',
            color='#9C4221', ha='center', va='center')
    ax.text(50.0, 15.6, "Alimenta o\nAntecedente\ndo Mamdani", fontsize=7.8,
            color='#7B341E', ha='center', va='center')
    ax.text(50.0, 11.2, "Rural: 0,75\nUrbano: 0,31\nFront: 0,18", fontsize=7.6,
            color='#C05621', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#FEEBC8', edgecolor='none', alpha=0.8))

    # Setas destacadas conectando Etapa 5 -> Caixa de Acoplamento -> Etapa 6
    draw_arrow(44.0, 14.5, 44.8, 14.5, color='#DD6B20', lw=2.5)
    draw_arrow(55.2, 14.5, 56.0, 14.5, color='#DD6B20', lw=2.5)

    save_figure(fig, "pipeline_integrado")


# ==============================================================================
# FIGURA 2: Trajetória dos Coeficientes Ridge vs lambda
# ==============================================================================
def generate_figure_2_ridge():
    print("Gerando Figura 2: Trajetória Ridge vs. lambda...")
    df = load_and_preprocess_pni(str(DATA_FILE))
    y_cont = np.log1p(df['dias_atraso'].values)
    y_bin = df['atraso_critico'].values

    feature_cols = ['sistema_offline', 'fabricante', 'faixa_etaria', 'mesmo_municipio', 'regiao']
    X_df = pd.get_dummies(df[feature_cols], drop_first=True)
    feature_names = list(X_df.columns)
    
    clean_feature_names = [
        f.replace("sistema_offline", "Sistema Offline")
         .replace("fabricante_", "Fab: ")
         .replace("faixa_etaria_", "Idade: ")
         .replace("mesmo_municipio", "Mesmo Município")
         .replace("regiao_", "Região: ")
        for f in feature_names
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_tr_cont, y_te_cont, y_tr_bin, y_te_bin = train_test_split(
        X_scaled, y_cont, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    N_tr, d = X_train.shape
    lambdas = np.logspace(-3, 3, 40)
    coef_trajectories = []
    r2_trajectories = []

    y_mean = np.mean(y_tr_cont)
    X_mean = np.mean(X_train, axis=0)
    XTX = X_train.T @ X_train
    XTy = X_train.T @ (y_tr_cont - y_mean)

    for lmb in lambdas:
        w = np.linalg.solve(XTX + N_tr * lmb * np.eye(d), XTy)
        b = y_mean - X_mean @ w
        coef_trajectories.append(w)
        pred = X_test @ w + b
        r2_trajectories.append(r2_score_scratch(y_te_cont, pred))

    coef_trajectories = np.array(coef_trajectories)
    r2_trajectories = np.array(r2_trajectories)
    best_idx = np.argmax(r2_trajectories)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    # Painel 1: Trajetória dos Coeficientes
    colormap = plt.cm.tab20(np.linspace(0, 1, d))
    for j in range(d):
        lw = 2.4 if 'offline' in feature_names[j] or 'mesmo_municipio' in feature_names[j] else 1.4
        axes[0].plot(lambdas, coef_trajectories[:, j],
                     label=clean_feature_names[j],
                     color=colormap[j], linewidth=lw, alpha=0.9)

    axes[0].set_xscale('log')
    axes[0].set_title(r'Trajetória de Encolhimento dos Coeficientes Ridge vs. $\lambda$', fontsize=11, fontweight='bold')
    axes[0].set_xlabel(r'Parâmetro de Regularização $\lambda$ (Escala Logarítmica)', fontsize=10.5)
    axes[0].set_ylabel(r'Magnitude dos Coeficientes $w_j$', fontsize=10.5)
    axes[0].axvline(lambdas[best_idx], color='crimson', linestyle='--', linewidth=1.8,
                    label=rf'$\lambda$ Ótimo ({lambdas[best_idx]:.3f})')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 0.98), fontsize=7.8, ncol=2, framealpha=0.92)
    axes[0].set_ylim(-0.25, 0.85)

    # Painel 2: R² no Teste vs lambda
    axes[1].plot(lambdas, r2_trajectories, color='forestgreen', linewidth=2.5, label='Desempenho $R^2$ (Teste)')
    axes[1].set_xscale('log')
    axes[1].set_title(r'Desempenho de Generalização ($R^2$) no Conjunto de Teste vs. $\lambda$', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(r'Parâmetro de Regularização $\lambda$ (Escala Logarítmica)', fontsize=10.5)
    axes[1].set_ylabel(r'Coeficiente de Determinação $R^2$', fontsize=10.5)
    axes[1].axvline(lambdas[best_idx], color='crimson', linestyle='--', linewidth=1.8,
                    label=rf'$\lambda$ Ótimo ({lambdas[best_idx]:.3f}) com $R^2 = {r2_trajectories[best_idx]:.4f}$')
    axes[1].scatter([lambdas[best_idx]], [r2_trajectories[best_idx]], color='crimson', s=60, zorder=5)
    axes[1].annotate(rf'$R^2_{{max}} = {r2_trajectories[best_idx]:.4f}$',
                    xy=(lambdas[best_idx], r2_trajectories[best_idx]),
                    xytext=(lambdas[best_idx] * 4, r2_trajectories[best_idx] - 0.04),
                    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                    fontsize=9.5, fontweight='bold')
    axes[1].set_ylim(0.0, 0.35)
    axes[1].legend(loc='lower left', fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    save_figure(fig, "etapa2_ridge_trajetoria")


# ==============================================================================
# FIGURA 3: Convergência do Algoritmo EM e Mistura Bimodal
# ==============================================================================
def generate_figure_3_em():
    print("Gerando Figura 3: Convergência do Algoritmo EM...")
    df = load_and_preprocess_pni(str(DATA_FILE))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['atraso_critico'])

    x_train_lat = np.log1p(train_df['dias_atraso'].values)
    em_model = ExpectationMaximizationGMM(K=2, max_iter=30, tol=1e-5)
    em_model.fit(x_train_lat)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=300)

    # Painel 1: Evolução Monótona da Log-Verossimilhança Marginal
    iters = np.arange(1, len(em_model.log_likelihood_history) + 1)
    ll_vals = em_model.log_likelihood_history

    axes[0].plot(iters, ll_vals, 'o-', color='indigo', linewidth=2.2, markersize=5.5, label=r'Log-Verossimilhança Marginal $\ln P(X \mid \Theta)$')
    axes[0].set_title(r'Convergência Estritamente Monótona do Algoritmo EM', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Iteração do Ciclo Expectation-Maximization ($t$)', fontsize=10.5)
    axes[0].set_ylabel(r'$\ln P(X \mid \Theta)$', fontsize=10.5)
    axes[0].axhline(ll_vals[-1], color='crimson', linestyle=':', label=f'Convergência final: {ll_vals[-1]:.2f}')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#FAF5FF', edgecolor='indigo', alpha=0.9)
    info_text = (
        r"$\mathcal{Q}(\Theta \mid \Theta^{(t)}) \geq \mathcal{Q}(\Theta^{(t)} \mid \Theta^{(t)})$" + "\n"
        f"Iterações totais: {len(ll_vals)}\n"
        f"L inicial: {ll_vals[0]:.1f}\n"
        f"L final: {ll_vals[-1]:.1f}\n"
        r"$\Delta \ln P(X \mid \Theta) \geq 0$ (Monotônico)"
    )
    axes[0].text(0.96, 0.22, info_text, transform=axes[0].transAxes, fontsize=8.8,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)
    axes[0].legend(loc='lower right', fontsize=9.2, framealpha=0.9)

    # Painel 2: Ajuste da Mistura Gaussiana 1D sobre Histograma de Latência
    count, bins, _ = axes[1].hist(x_train_lat, bins=45, density=True, alpha=0.35, color='gray', edgecolor='white', label='Microdados PNI (Treino)')
    grid_x = np.linspace(bins[0], bins[-1], 300)

    pdf_0 = em_model.theta[0] * em_model._gaussian_pdf(grid_x, em_model.mu[0], em_model.sigma2[0])
    pdf_1 = em_model.theta[1] * em_model._gaussian_pdf(grid_x, em_model.mu[1], em_model.sigma2[1])
    pdf_total = pdf_0 + pdf_1

    dias_0 = np.expm1(em_model.mu[0])
    dias_1 = np.expm1(em_model.mu[1])

    axes[1].plot(grid_x, pdf_0, '--', color='darkcyan', linewidth=2.0,
                 label=rf'Regime Ágil $Z=0$ ($\pi_0={em_model.theta[0]:.2f}, \mu_0={em_model.mu[0]:.2f} \approx {dias_0:.1f}$ dias)')
    axes[1].plot(grid_x, pdf_1, '--', color='darkorange', linewidth=2.0,
                 label=rf'Regime Represado $Z=1$ ($\pi_1={em_model.theta[1]:.2f}, \mu_1={em_model.mu[1]:.2f} \approx {dias_1:.1f}$ dias)')
    axes[1].plot(grid_x, pdf_total, '-', color='darkred', linewidth=2.5, label=r'Densidade da Mistura Bimodal $\sum \pi_k \mathcal{N}$')

    axes[1].set_title(r'Separação dos Regimes Operacionais Latentes $Z \in \{0, 1\}$', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(r'Latência de Notificação: $\ln(1 + \mathrm{dias\_atraso})$', fontsize=10.5)
    axes[1].set_ylabel('Densidade de Probabilidade', fontsize=10.5)
    axes[1].legend(loc='upper right', fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    save_figure(fig, "etapa4_em_convergencia")


# ==============================================================================
# FIGURA 4: Rede Bayesiana Causal (DAG, Nó Colisor & Economia de Parâmetros)
# ==============================================================================
def generate_figure_4_bayesian_dag():
    print("Gerando Figura 4: DAG Causal da Rede Bayesiana...")
    fig = plt.figure(figsize=(14.5, 5.5), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.2, 0.8])

    # Subplot 1 (Left, Spanning both rows): Visualização do DAG
    ax_dag = fig.add_subplot(gs[:, 0])
    ax_dag.axis('off')

    edges = [
        ('R', 'S'),
        ('S', 'A'),
        ('V', 'N'),
        ('N', 'A')
    ]
    G = nx.DiGraph(edges)
    
    pos = {
        'R': (1.4, 4.1),
        'V': (6.6, 4.1),
        'S': (2.3, 2.4),
        'N': (5.7, 2.4),
        'A': (4.0, 0.7)
    }

    regular_nodes = ['R', 'V', 'S', 'N']
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=5200,
                           node_color='#EBF8FF', edgecolors='#2B6CB0', linewidths=2.2, ax=ax_dag)

    # Destaque ao Nó Colisor A
    nx.draw_networkx_nodes(G, pos, nodelist=['A'], node_size=6500,
                           node_color='#FFF5F5', edgecolors='#C53030', linewidths=3.2, ax=ax_dag)

    nx.draw_networkx_edges(G, pos, edgelist=[('R', 'S'), ('V', 'N')],
                           arrowstyle='->', arrowsize=22, edge_color='#4A5568', width=2.0, ax=ax_dag)
    nx.draw_networkx_edges(G, pos, edgelist=[('S', 'A'), ('N', 'A')],
                           arrowstyle='->', arrowsize=25, edge_color='#C53030', width=2.8, ax=ax_dag)

    node_labels = {
        'R': r"$\mathbf{R}$ (Região)" + "\n" + r"$\mathrm{N/NE \ vs \ S/SE/CO}$",
        'V': r"$\mathbf{V}$ (Faixa Etária)" + "\n" + r"$\mathrm{Vuln \ vs \ Adulto}$",
        'S': r"$\mathbf{S}$ (Modo Sistema)" + "\n" + r"$\mathrm{Offline \ vs \ Online}$",
        'N': r"$\mathbf{N}$ (Fabricante)" + "\n" + r"$\mathrm{Nac \ vs \ Mult}$",
        'A': r"$\mathbf{A}$ (Atraso Crítico)" + "\n" + r"$\mathbf{> 7 \ dias \ [COLISOR]}$"
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8.0, font_family='sans-serif', ax=ax_dag)

    edge_notes = {
        (1.50, 3.25): "Disparidade\ndigital",
        (6.50, 3.25): "Diretriz\ncalendário",
        (2.75, 1.45): "Digitação local\ne sincronização",
        (5.25, 1.45): "Logística e\nconferência"
    }
    for (x, y), text in edge_notes.items():
        ax_dag.text(x, y, text, fontsize=7.8, color='#4A5568', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#F7FAFC', edgecolor='#CBD5E0', alpha=0.9))

    ax_dag.set_title("Grafo Acíclico Dirigido (DAG) Causal & Nó Colisor $A$", fontsize=11.5, fontweight='bold', pad=12)
    ax_dag.set_xlim(0.4, 7.6)
    ax_dag.set_ylim(-0.2, 4.8)

    # Subplot 2 (Right Top): Análise Formal de d-Separação
    ax_dsep = fig.add_subplot(gs[0, 1])
    ax_dsep.axis('off')
    ax_dsep.set_xlim(0, 1)
    ax_dsep.set_ylim(0, 1)

    card_dsep = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.03,rounding_size=0.04",
                               facecolor='#FFFDF5', edgecolor='#D69E2E', linewidth=1.6)
    ax_dsep.add_patch(card_dsep)
    ax_dsep.text(0.05, 0.88, "Propriedades Formais de d-Separação", fontsize=10.5, fontweight='bold', color='#744210', va='top')

    dsep_lines = [
        r"1. Cadeia Causal $R \to S \to A$:",
        r"    $\bullet \ (R \perp\!\!\perp A \mid S) \ \Rightarrow \ \mathbf{Verdadeiro}$ (Observar $S$ bloqueia $R$)",
        r"2. Colisor Incondicionado $S \to A \leftarrow N$:",
        r"    $\bullet \ (S \perp\!\!\perp N) \ \Rightarrow \ \mathbf{Verdadeiro}$ (Marginalmente independentes)",
        r"3. Colisor Condicionado no Nó $A$:",
        r"    $\bullet \ (S \not\perp N \mid A) \ \Rightarrow \ \mathbf{Falso}$ ($\mathbf{Caminho \ Ativado!}$)",
        r"    $\rightarrow$ Fenômeno de Explaining Away: evidência de $N$ altera crença em $S$."
    ]
    curr_y = 0.74
    for l in dsep_lines:
        ax_dsep.text(0.05, curr_y, l, fontsize=8.2, color='#2D3748', va='top')
        curr_y -= 0.125

    # Subplot 3 (Right Bottom): Gráfico de Barras de Economia de Parâmetros
    ax_bar = fig.add_subplot(gs[1, 1])
    y_pos = [0, 1]
    param_counts = [10, 31]
    labels_bar = ['Rede Bayesiana\n(Fatorada)', 'Tabela Conjunta\nCompleta ($2^5 - 1$)']

    bars = ax_bar.barh(y_pos, param_counts, color=['#38A169', '#E53E3E'],
                       height=0.55, alpha=0.88, edgecolor='black', linewidth=1.0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels_bar, fontsize=8.8, fontweight='bold')
    ax_bar.set_xlim(0, 38)
    ax_bar.set_xlabel('Número de Parâmetros Independentes', fontsize=9.2)
    ax_bar.set_title(r'Economia Estrutural: 67,74% de Redução (21 parâmetros)', fontsize=10.2, fontweight='bold', color='#22543D')

    ax_bar.text(10 + 0.8, 0, '10 parâmetros', va='center', fontweight='bold', color='#22543D', fontsize=8.8)
    ax_bar.text(31 + 0.8, 1, '31 parâmetros', va='center', fontweight='bold', color='#9B2C2C', fontsize=8.8)

    plt.tight_layout()
    save_figure(fig, "etapa5_rede_bayesiana_dag")


# ==============================================================================
# FIGURA 5: Sistema Fuzzy Mamdani (Pertinências, Fronteira & Superfície 3D)
# ==============================================================================
def generate_figure_5_fuzzy_surface():
    print("Gerando Figura 5: Superfície de Controle Fuzzy e Decisão Suave...")
    fuzzy_sys = HealthFuzzyInferenceSystem()

    fig = plt.figure(figsize=(15.0, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    # 1. Pertinência: Risco Bayesiano
    ax_r = fig.add_subplot(gs[0, 0])
    u_risco = fuzzy_sys.risco.universe
    ax_r.plot(u_risco, fuzzy_sys.risco['baixo'].mf, color='forestgreen', linewidth=2.0, label='Baixo')
    ax_r.plot(u_risco, fuzzy_sys.risco['medio'].mf, color='darkorange', linewidth=2.0, label='Médio')
    ax_r.plot(u_risco, fuzzy_sys.risco['alto'].mf, color='crimson', linewidth=2.0, label='Alto')
    ax_r.set_title('(a) Antecedente 1: Risco Bayesiano', fontsize=10.5, fontweight='bold')
    ax_r.set_xlabel(r'Probabilidade $P(A=1 \mid \mathbf{e})$', fontsize=9.5)
    ax_r.set_ylabel(r'Grau de Pertinência $\mu$', fontsize=9.5)
    ax_r.legend(loc='upper right', fontsize=8.5)
    ax_r.set_ylim(-0.05, 1.05)

    # 2. Pertinência: Severidade/Vulnerabilidade
    ax_v = fig.add_subplot(gs[0, 1])
    u_vuln = fuzzy_sys.vulnerabilidade.universe
    ax_v.plot(u_vuln, fuzzy_sys.vulnerabilidade['baixa'].mf, color='forestgreen', linewidth=2.0, label='Baixa')
    ax_v.plot(u_vuln, fuzzy_sys.vulnerabilidade['media'].mf, color='darkorange', linewidth=2.0, label='Média')
    ax_v.plot(u_vuln, fuzzy_sys.vulnerabilidade['alta'].mf, color='crimson', linewidth=2.0, label='Alta')
    ax_v.set_title('(b) Antecedente 2: Vulnerabilidade', fontsize=10.5, fontweight='bold')
    ax_v.set_xlabel('% População Prioritária (Crianças/Idosos)', fontsize=9.5)
    ax_v.set_ylabel(r'Grau de Pertinência $\mu$', fontsize=9.5)
    ax_v.legend(loc='upper right', fontsize=8.5)
    ax_v.set_ylim(-0.05, 1.05)

    # 3. Pertinência: Urgência Consequente
    ax_u = fig.add_subplot(gs[0, 2])
    u_urg = fuzzy_sys.urgencia.universe
    ax_u.plot(u_urg, fuzzy_sys.urgencia['baixa'].mf, color='forestgreen', linewidth=2.0, label='Baixa')
    ax_u.plot(u_urg, fuzzy_sys.urgencia['media'].mf, color='darkorange', linewidth=2.0, label='Média')
    ax_u.plot(u_urg, fuzzy_sys.urgencia['alta'].mf, color='crimson', linewidth=2.0, label='Alta')
    ax_u.set_title('(c) Consequente: Urgência SUS', fontsize=10.5, fontweight='bold')
    ax_u.set_xlabel('Escore de Urgência de Intervenção [0 - 100]', fontsize=9.5)
    ax_u.set_ylabel(r'Grau de Pertinência $\mu$', fontsize=9.5)
    ax_u.legend(loc='upper right', fontsize=8.5)
    ax_u.set_ylim(-0.05, 1.05)

    # 4. Decisão Suave vs Limiar Rígido nos Cenários SUS
    ax_comp = fig.add_subplot(gs[1, :2])
    riscos_grid = np.linspace(0.0, 1.0, 100)
    decisoes_fuzzy_75 = [fuzzy_sys.evaluate(r, 75.0) for r in riscos_grid]
    decisoes_fuzzy_50 = [fuzzy_sys.evaluate(r, 50.0) for r in riscos_grid]
    decisoes_fuzzy_25 = [fuzzy_sys.evaluate(r, 25.0) for r in riscos_grid]
    decisoes_hard = [80.0 if r > 0.5 else 20.0 for r in riscos_grid]

    ax_comp.plot(riscos_grid, decisoes_fuzzy_75, color='darkblue', linewidth=2.5, label='Fuzzy Mamdani (Vuln = 75%)')
    ax_comp.plot(riscos_grid, decisoes_fuzzy_50, color='royalblue', linestyle='-.', linewidth=1.8, label='Fuzzy Mamdani (Vuln = 50%)')
    ax_comp.plot(riscos_grid, decisoes_fuzzy_25, color='cornflowerblue', linestyle=':', linewidth=1.8, label='Fuzzy Mamdani (Vuln = 25%)')
    ax_comp.plot(riscos_grid, decisoes_hard, color='crimson', linestyle='--', linewidth=2.2, label='Limiar Booleano Rígido (Corte 0.50)')
    ax_comp.axvline(0.5, color='gray', linestyle=':', alpha=0.7)

    # Destaque dos 3 Cenários SUS sem sobreposição
    cenarios = [
        ('Rural Remoto\n(P=0.75, V=85% -> 64.61)', 0.7458, 64.61, 'darkred', (-65, 18)),
        ('Urbano Central\n(P=0.31, V=30% -> 38.79)', 0.3059, 38.79, 'darkgreen', (22, -34)),
        ('Fronteira\n(P=0.18, V=65% -> 23.90)', 0.1758, 23.90, 'darkorange', (-80, 18))
    ]
    for label, r_val, u_val, col, offset in cenarios:
        ax_comp.scatter([r_val], [u_val], color=col, s=75, zorder=6, edgecolors='black')
        ax_comp.annotate(label, xy=(r_val, u_val),
                         xytext=(offset[0], offset[1]), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                         fontsize=8.5, fontweight='bold', color=col,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=col, alpha=0.9))

    ax_comp.set_title('(d) Comparação na Fronteira: Transição Suave Fuzzy vs. Degrau Booleano Rígido', fontsize=11, fontweight='bold')
    ax_comp.set_xlabel(r'Risco Bayesiano de Atraso Crítico $P(A=1 \mid \mathbf{e})$', fontsize=10.5)
    ax_comp.set_ylabel('Escore de Prioridade / Urgência SUS [0 - 100]', fontsize=10.5)
    ax_comp.legend(loc='upper left', fontsize=8.8, framealpha=0.92)
    ax_comp.set_ylim(10, 90)

    # 5. Superfície 3D de Controle
    ax_3d = fig.add_subplot(gs[1, 2], projection='3d')
    X_surf, Y_surf, Z_surf = fuzzy_sys.compute_control_surface(resolution=28)
    surf = ax_3d.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', edgecolor='none', alpha=0.88)
    ax_3d.set_title('(e) Superfície de Controle 3D', fontsize=11, fontweight='bold', pad=12)
    ax_3d.set_xlabel(r'$P(\mathrm{Atraso})$', fontsize=9.0, labelpad=5)
    ax_3d.set_ylabel('Vuln (%)', fontsize=9.0, labelpad=5)
    ax_3d.set_zlabel('Urgência', fontsize=9.0, labelpad=5)
    ax_3d.view_init(elev=26, azim=225)
    fig.colorbar(surf, ax=ax_3d, shrink=0.45, aspect=10, pad=0.1, label='Urgência Final')

    plt.tight_layout()
    save_figure(fig, "etapa6_fuzzy_superficie")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print("==================================================================")
    print("Exportação Científica de Figuras em Alta Resolução (SBC / PNI)")
    print(f"Diretório de Destino: {OUTPUT_DIR}")
    print("==================================================================")

    generate_figure_1_pipeline()
    generate_figure_2_ridge()
    generate_figure_3_em()
    generate_figure_4_bayesian_dag()
    generate_figure_5_fuzzy_surface()

    print("\nVerificando integridade dos arquivos gerados em 'figures/'...")
    expected_files = [
        "pipeline_integrado.pdf", "pipeline_integrado.png",
        "etapa2_ridge_trajetoria.pdf", "etapa2_ridge_trajetoria.png",
        "etapa4_em_convergencia.pdf", "etapa4_em_convergencia.png",
        "etapa5_rede_bayesiana_dag.pdf", "etapa5_rede_bayesiana_dag.png",
        "etapa6_fuzzy_superficie.pdf", "etapa6_fuzzy_superficie.png"
    ]
    all_ok = True
    for f in expected_files:
        p = OUTPUT_DIR / f
        if p.exists() and p.stat().st_size > 0:
            print(f"  [PASS] {f} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [FAIL] {f} (Ausente ou vazio!)")
            all_ok = False

    if all_ok:
        print("\nSucesso total: Todas as 5 figuras científicas (.pdf e .png) foram geradas com sucesso!")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
