"""
Implementação de Rede Bayesiana e Inferência por Enumeração do Zero:
- Grafo Acíclico Dirigido (DAG) para o Domínio de Saúde Pública
- Estimação de Tabelas de Probabilidade Condicional (CPTs)
- Algoritmo de Inferência Exata por Enumeração (Russell & Norvig, Cap. 14)
- Análise de d-separação e Economia de Parâmetros
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Any, Optional

class BayesianNetworkScratch:
    """
    Rede Bayesiana com 5 variáveis binárias para o domínio de Saúde Pública:
    R: Região Norte/Nordeste (0: Outras, 1: N/NE)
    S: Sistema Offline (0: Online, 1: Offline)
    V: Faixa Etária Vulnerável Criança/Idoso (0: Jovem/Adulto, 1: Vulnerável)
    N: Fabricante Nacional Butantan/Fiocruz (0: Outro, 1: Nacional)
    A: Atraso Crítico de Notificação (0: Oportuno, 1: Tardia)
    
    Fatoração da Conjunta segundo o DAG:
    P(R, S, V, N, A) = P(R) * P(S | R) * P(V) * P(N | V) * P(A | S, N)
    """
    def __init__(self):
        self.variables = ['R', 'S', 'V', 'N', 'A']
        self.cpt_R: Dict[int, float] = {}
        self.cpt_S_R: Dict[int, Dict[int, float]] = {}
        self.cpt_V: Dict[int, float] = {}
        self.cpt_N_V: Dict[int, Dict[int, float]] = {}
        self.cpt_A_SN: Dict[Tuple[int, int], Dict[int, float]] = {}
        
    def fit(self, df: pd.DataFrame):
        """Estima as CPTs a partir dos microdados empíricos usando contagem de frequências (MLE)."""
        N_total = len(df)
        
        # 1. P(R)
        r_counts = df['R'].value_counts()
        for r in [0, 1]:
            self.cpt_R[r] = float(r_counts.get(r, 0) / N_total)
            
        # 2. P(S | R)
        for r in [0, 1]:
            sub_r = df[df['R'] == r]
            s_counts = sub_r['S'].value_counts()
            n_r = max(len(sub_r), 1)
            self.cpt_S_R[r] = {s: float(s_counts.get(s, 0) / n_r) for s in [0, 1]}
            
        # 3. P(V)
        v_counts = df['V'].value_counts()
        for v in [0, 1]:
            self.cpt_V[v] = float(v_counts.get(v, 0) / N_total)
            
        # 4. P(N | V)
        for v in [0, 1]:
            sub_v = df[df['V'] == v]
            n_counts = sub_v['N'].value_counts()
            n_v = max(len(sub_v), 1)
            self.cpt_N_V[v] = {n: float(n_counts.get(n, 0) / n_v) for n in [0, 1]}
            
        # 5. P(A | S, N) - Nó colisor / dependente de S e N
        for s in [0, 1]:
            for n in [0, 1]:
                sub_sn = df[(df['S'] == s) & (df['N'] == n)]
                a_counts = sub_sn['A'].value_counts()
                n_sn = max(len(sub_sn), 1)
                self.cpt_A_SN[(s, n)] = {a: float(a_counts.get(a, 0) / n_sn) for a in [0, 1]}
                
        return self
        
    def joint_probability(self, r: int, s: int, v: int, n: int, a: int) -> float:
        """Calcula P(R=r, S=s, V=v, N=n, A=a) pelo produto das CPTs."""
        p_r = self.cpt_R[r]
        p_s = self.cpt_S_R[r][s]
        p_v = self.cpt_V[v]
        p_n = self.cpt_N_V[v][n]
        p_a = self.cpt_A_SN[(s, n)][a]
        return p_r * p_s * p_v * p_n * p_a
        
    def query_by_enumeration(self, query_var: str, evidence: Dict[str, int]) -> Dict[int, float]:
        """
        Algoritmo de Inferência Exata por Enumeração (Russell & Norvig, AIMA Cap. 14).
        P(X | e) = alpha * sum_Y P(X, e, Y), onde Y sao as variaveis ocultas.
        """
        distribution = {0: 0.0, 1: 0.0}
        
        # Variáveis ocultas = todas as variáveis que não são query nem evidência
        hidden_vars = [var for var in self.variables if var != query_var and var not in evidence]
        
        for q_val in [0, 1]:
            total_prob = 0.0
            
            # Iterar sobre todas as combinações de valores das variáveis ocultas {0, 1}^|Y|
            for h_comb in product([0, 1], repeat=len(hidden_vars)):
                state = dict(evidence)
                state[query_var] = q_val
                for h_var, h_val in zip(hidden_vars, h_comb):
                    state[h_var] = h_val
                    
                prob = self.joint_probability(
                    r=state['R'], s=state['S'], v=state['V'], n=state['N'], a=state['A']
                )
                total_prob += prob
                
            distribution[q_val] = total_prob
            
        # Normalização alpha = 1 / sum(distribution.values())
        norm_factor = sum(distribution.values())
        if norm_factor > 0:
            for k in distribution:
                distribution[k] /= norm_factor
                
        return distribution
        
    def calculate_parameter_economy(self) -> Dict[str, Any]:
        """Calcula a redução de parâmetros da rede Bayesiana em relação à tabela conjunta completa."""
        n_vars = 5
        # Tabela conjunta completa: 2^n - 1 parâmetros independentes
        full_joint_params = (2 ** n_vars) - 1
        
        # Parâmetros independentes da Rede Bayesiana:
        # P(R): 1 (já que P(R=0) = 1 - P(R=1))
        # P(S | R): 2 (um para R=0, outro para R=1)
        # P(V): 1
        # P(N | V): 2 (um para V=0, outro para V=1)
        # P(A | S, N): 4 (um para cada combinação de (S, N) in {0,1}^2)
        bn_params = 1 + 2 + 1 + 2 + 4  # Total: 10 parâmetros!
        
        economy = full_joint_params - bn_params
        reduction_pct = (economy / full_joint_params) * 100.0
        
        return {
            "num_variables": n_vars,
            "full_joint_parameters": full_joint_params,
            "bayesian_network_parameters": bn_params,
            "parameters_saved": economy,
            "reduction_percentage": reduction_pct
        }
