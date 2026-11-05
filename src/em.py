"""
Implementações para a Etapa 4:
1. MLE e MAP para distribuição de Bernoulli com prior conjugada Beta
2. Naive Bayes do zero (Categorical Naive Bayes com Laplace Smoothing)
3. Algoritmo Expectation-Maximization (EM) para Modelo de Mistura com Variável Oculta Z
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class BernoulliEstimator:
    """Estimadores MLE e MAP para o parâmetro theta de uma Bernoulli."""
    @staticmethod
    def mle(k: int, n: int) -> float:
        """theta_MLE = k / n"""
        return float(k / max(n, 1))
        
    @staticmethod
    def map(k: int, n: int, alpha: float = 2.0, beta: float = 2.0) -> float:
        """theta_MAP com prior Beta(alpha, beta) = (k + alpha - 1) / (n + alpha + beta - 2)"""
        denom = n + alpha + beta - 2.0
        return float((k + alpha - 1.0) / max(denom, 1e-12))

class NaiveBayesScratch:
    """
    Classificador Naive Bayes categórico implementado do zero.
    P(y | x) propto P(y) * prod_j P(x_j | y)
    Aplica suavização de Laplace para prevenir probabilidades nulas.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Fator de suavização de Laplace
        self.class_priors: Dict[int, float] = {}
        # feature_probs[feature_name][class_val][feature_val] = prob
        self.feature_probs: Dict[str, Dict[int, Dict[any, float]]] = {}
        self.classes: np.ndarray = np.array([])
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y).astype(int)
        self.classes = np.unique(y)
        N = len(y)
        
        self.class_priors = {}
        for c in self.classes:
            self.class_priors[c] = float((np.sum(y == c) + self.alpha) / (N + len(self.classes) * self.alpha))
            
        self.feature_probs = {}
        for col in X.columns:
            self.feature_probs[col] = {}
            unique_vals = X[col].unique()
            V = len(unique_vals)
            
            for c in self.classes:
                self.feature_probs[col][c] = {}
                subset_col = X.loc[y == c, col]
                n_c = len(subset_col)
                counts = subset_col.value_counts().to_dict()
                
                for val in unique_vals:
                    val_count = counts.get(val, 0)
                    prob = (val_count + self.alpha) / (n_c + V * self.alpha)
                    self.feature_probs[col][c][val] = float(prob)
                    
        return self
        
    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_probs = []
        for _, row in X.iterrows():
            row_scores = []
            for c in self.classes:
                # log P(c)
                score = np.log(self.class_priors[c])
                for col in X.columns:
                    val = row[col]
                    prob_val = self.feature_probs[col][c].get(val, 1e-6)
                    score += np.log(prob_val)
                row_scores.append(score)
            log_probs.append(row_scores)
            
        log_probs = np.array(log_probs)
        # Normalização log-sum-exp para probabilidades posteriores [0, 1]
        max_scores = np.max(log_probs, axis=1, keepdims=True)
        exp_scores = np.exp(log_probs - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_log_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

class ExpectationMaximizationGMM:
    """
    Algoritmo EM para Modelo de Mistura Gaussiana 1D com K componentes latentes Z in {0, 1}.
    Exatamente conforme o pseudocódigo do Listing 3 da especificação do curso.
    """
    def __init__(self, K: int = 2, max_iter: int = 50, tol: float = 1e-5):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.theta: np.ndarray = np.zeros(K)      # Pesos da mistura P(Z=k)
        self.mu: np.ndarray = np.zeros(K)         # Médias mu_k
        self.sigma2: np.ndarray = np.zeros(K)     # Variâncias sigma_k^2
        self.responsibilities: Optional[np.ndarray] = None
        self.log_likelihood_history: List[float] = []
        
    def _gaussian_pdf(self, x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
        var = max(sigma2, 1e-4)
        denom = np.sqrt(2.0 * np.pi * var)
        exponent = -0.5 * ((x - mu) ** 2) / var
        return (1.0 / denom) * np.exp(np.clip(exponent, -50.0, 50.0))
        
    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        
        # Inicialização baseada em quantis para convergência estável
        self.theta = np.ones(self.K) / self.K
        quantiles = np.linspace(0.2, 0.8, self.K)
        self.mu = np.quantile(x, quantiles)
        total_var = np.var(x)
        self.sigma2 = np.full(self.K, max(total_var / 2.0, 1.0))
        
        self.log_likelihood_history = []
        r = np.zeros((self.K, N))
        
        for it in range(self.max_iter):
            # ===== E-STEP =====
            for k in range(self.K):
                r[k, :] = self._gaussian_pdf(x, self.mu[k], self.sigma2[k]) * self.theta[k]
                
            # Normalizar r[:, i]
            total_r = np.sum(r, axis=0, keepdims=True)
            r = r / np.maximum(total_r, 1e-12)
            
            # ===== M-STEP =====
            Nk = np.sum(r, axis=1)  # shape (K,)
            for k in range(self.K):
                self.theta[k] = Nk[k] / N
                self.mu[k] = np.sum(r[k, :] * x) / max(Nk[k], 1e-12)
                diff_sq = (x - self.mu[k]) ** 2
                self.sigma2[k] = np.sum(r[k, :] * diff_sq) / max(Nk[k], 1e-12)
                self.sigma2[k] = max(self.sigma2[k], 1e-4)  # Regularização contra singularidades
                
            # ===== Log-Verossimilhança =====
            # L = sum_i log(sum_k theta_k * N(x_i | mu_k, sigma_k^2))
            mixture_densities = np.zeros(N)
            for k in range(self.K):
                mixture_densities += self.theta[k] * self._gaussian_pdf(x, self.mu[k], self.sigma2[k])
                
            log_L = np.sum(np.log(np.maximum(mixture_densities, 1e-12)))
            self.log_likelihood_history.append(float(log_L))
            
            # Checagem de convergência
            if it > 0 and abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < self.tol:
                break
                
        self.responsibilities = r.T  # shape (N, K)
        return self
        
    def predict_latent(self, x: np.ndarray) -> np.ndarray:
        """Infere o componente latente mais provável z = argmax_k r_k."""
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        r = np.zeros((self.K, N))
        for k in range(self.K):
            r[k, :] = self._gaussian_pdf(x, self.mu[k], self.sigma2[k]) * self.theta[k]
        return np.argmax(r, axis=0)
