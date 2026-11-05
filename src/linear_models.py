"""
Implementação de Modelos Lineares do Zero:
1. Regressão Linear Univariada (Mínimos Quadrados / Fórmula Fechada)
2. Regressão Linear Multivariada com Gradiente Descendente e Regularização L2 (Ridge)
3. Regressão Logística com Gradiente Descendente para Classificação Binária
"""

import numpy as np
from typing import Tuple, Optional, List

class LinearRegressionOLS:
    """Regressão Linear Univariada com Solução Analítica (Fórmula Fechada)."""
    def __init__(self):
        self.beta_0: float = 0.0
        self.beta_1: float = 0.0
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        
        self.beta_1 = float(cov_xy / max(var_x, 1e-12))
        self.beta_0 = float(y_mean - self.beta_1 * x_mean)
        return self
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.beta_0 + self.beta_1 * x

class LinearRegressionGD:
    """
    Regressão Linear Multivariada com Gradiente Descendente e Regularização L2 (Ridge).
    Minimiza: J(w) = (1 / 2N) * sum((y - Xw)^2) + (lambda / 2) * sum(w[1:]^2)
    """
    def __init__(self, learning_rate: float = 0.05, num_iter: int = 1000, l2_lambda: float = 0.0):
        self.lr = learning_rate
        self.num_iter = num_iter
        self.l2_lambda = l2_lambda
        self.w: Optional[np.ndarray] = None
        self.cost_history: List[float] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N, d = X.shape
        
        # Adicionar coluna de bias (intercepto)
        X_b = np.hstack([np.ones((N, 1)), X])
        
        # Inicializar pesos w = zeros(d+1)
        self.w = np.zeros(d + 1)
        self.cost_history = []
        
        for _ in range(self.num_iter):
            preds = np.dot(X_b, self.w)
            errors = preds - y  # (y_pred - y_true)
            
            # Gradiente MSE
            grad = np.dot(X_b.T, errors) / N
            
            # Penalidade L2 (Ridge) apenas nos coeficientes w[1:], não no bias w[0]
            if self.l2_lambda > 0:
                l2_term = np.zeros_like(self.w)
                l2_term[1:] = self.l2_lambda * self.w[1:]
                grad += l2_term
                
            # Atualização dos pesos
            self.w -= self.lr * grad
            
            # Custo MSE
            cost = (1.0 / (2.0 * N)) * np.sum(errors ** 2)
            if self.l2_lambda > 0:
                cost += (self.l2_lambda / 2.0) * np.sum(self.w[1:] ** 2)
            self.cost_history.append(float(cost))
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("O modelo precisa ser treinado com fit() antes de predict().")
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        X_b = np.hstack([np.ones((N, 1)), X])
        return np.dot(X_b, self.w)

class LogisticRegressionGD:
    """
    Regressão Logística Binária com Sigmoide e Gradiente Descendente do zero.
    p = 1 / (1 + exp(-z)), z = Xw
    """
    def __init__(self, learning_rate: float = 0.05, num_iter: int = 1000):
        self.lr = learning_rate
        self.num_iter = num_iter
        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Prevenção contra overflow numérico
        z_clipped = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N, d = X.shape
        
        X_b = np.hstack([np.ones((N, 1)), X])
        self.w = np.zeros(d + 1)
        self.loss_history = []
        
        for _ in range(self.num_iter):
            z = np.dot(X_b, self.w)
            probs = self.sigmoid(z)
            
            # Gradiente da Log-Loss: (1/N) * X^T (probs - y)
            errors = probs - y
            grad = np.dot(X_b.T, errors) / N
            self.w -= self.lr * grad
            
            # Binary Cross Entropy Loss
            eps = 1e-12
            loss = - (1.0 / N) * np.sum(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
            self.loss_history.append(float(loss))
            
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Modelo não treinado.")
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        X_b = np.hstack([np.ones((N, 1)), X])
        return self.sigmoid(np.dot(X_b, self.w))
        
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o coeficiente de determinação R² = 1 - (SS_res / SS_tot)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / max(ss_tot, 1e-12)))

def mse_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))
