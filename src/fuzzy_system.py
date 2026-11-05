"""
Sistema de Inferência Fuzzy (Mamdani) para a Etapa 6:
Combina o Risco Bayesiano da Etapa 5 e a Vulnerabilidade da População
em um Índice Suave de Urgência de Intervenção em Saúde Pública.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Tuple, Dict, Any

class HealthFuzzyInferenceSystem:
    def __init__(self):
        # 1. Variáveis Linguísticas
        # Antecedente 1: Risco Bayesiano de Atraso Crítico [0.0, 1.0]
        self.risco = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'risco')
        
        # Antecedente 2: Índice de Vulnerabilidade Populacional [0, 100]
        self.vulnerabilidade = ctrl.Antecedent(np.arange(0, 101, 1), 'vulnerabilidade')
        
        # Consequente: Urgência de Auditoria e Apoio Técnico [0, 100]
        self.urgencia = ctrl.Consequent(np.arange(0, 101, 1), 'urgencia')
        
        # 2. Funções de Pertinência (Triangulares)
        self.risco['baixo'] = fuzz.trimf(self.risco.universe, [0.0, 0.0, 0.4])
        self.risco['medio'] = fuzz.trimf(self.risco.universe, [0.2, 0.5, 0.8])
        self.risco['alto']  = fuzz.trimf(self.risco.universe, [0.6, 1.0, 1.0])
        
        self.vulnerabilidade['baixa'] = fuzz.trimf(self.vulnerabilidade.universe, [0, 0, 40])
        self.vulnerabilidade['media'] = fuzz.trimf(self.vulnerabilidade.universe, [20, 50, 80])
        self.vulnerabilidade['alta']  = fuzz.trimf(self.vulnerabilidade.universe, [60, 100, 100])
        
        self.urgencia['baixa'] = fuzz.trimf(self.urgencia.universe, [0, 0, 40])
        self.urgencia['media'] = fuzz.trimf(self.urgencia.universe, [25, 50, 75])
        self.urgencia['alta']  = fuzz.trimf(self.urgencia.universe, [60, 100, 100])
        
        # 3. Base de Regras Especialistas (Mamdani)
        r1 = ctrl.Rule(self.risco['alto'] & self.vulnerabilidade['alta'], self.urgencia['alta'])
        r2 = ctrl.Rule(self.risco['alto'] & ~self.vulnerabilidade['alta'], self.urgencia['media'])
        r3 = ctrl.Rule(self.risco['medio'], self.urgencia['media'])
        r4 = ctrl.Rule(self.risco['baixo'] & ~self.vulnerabilidade['alta'], self.urgencia['baixa'])
        r5 = ctrl.Rule(self.risco['baixo'] & self.vulnerabilidade['alta'], self.urgencia['media'])
        
        self.system = ctrl.ControlSystem([r1, r2, r3, r4, r5])
        self.simulation = ctrl.ControlSystemSimulation(self.system)
        
    def evaluate(self, risco_val: float, vulnerabilidade_val: float) -> float:
        """Calcula a urgência de intervenção para um dado par de entradas."""
        self.simulation.input['risco'] = float(np.clip(risco_val, 0.0, 1.0))
        self.simulation.input['vulnerabilidade'] = float(np.clip(vulnerabilidade_val, 0.0, 100.0))
        self.simulation.compute()
        return float(self.simulation.output['urgencia'])
        
    def compute_control_surface(self, resolution: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gera a superfície de controle 3D (Risco x Vulnerabilidade -> Urgência)."""
        x_risco = np.linspace(0.0, 1.0, resolution)
        y_vuln = np.linspace(0.0, 100.0, resolution)
        X, Y = np.meshgrid(x_risco, y_vuln)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.evaluate(X[i, j], Y[i, j])
                
        return X, Y, Z
