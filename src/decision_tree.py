"""
Implementação do algoritmo LEARN-DECISION-TREE do zero.
Baseado em Russell & Norvig (AIMA Cap. 18) e no pseudocódigo da Etapa 1.
Suporta cálculo exato de entropia, ganho de informação, poda por profundidade máxima,
validação cruzada e inspeção visual da árvore.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

def entropy(y: np.ndarray) -> float:
    """Calcula a entropia de Shannon de uma distribuição de classes H(S) = - sum(p * log2(p))."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Evitar log2(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def information_gain(df: pd.DataFrame, attribute: str, target: str) -> float:
    """Calcula o Ganho de Informação IG(S, A) = H(S) - sum(|Sv|/|S| * H(Sv))."""
    if len(df) == 0:
        return 0.0
    
    h_parent = entropy(df[target].values)
    values = df[attribute].unique()
    h_children = 0.0
    n = len(df)
    
    for val in values:
        subset = df[df[attribute] == val]
        weight = len(subset) / n
        h_children += weight * entropy(subset[target].values)
        
    return float(h_parent - h_children)

class DecisionTreeNode:
    def __init__(
        self,
        attribute: Optional[str] = None,
        children: Optional[Dict[Any, 'DecisionTreeNode']] = None,
        leaf_value: Optional[Any] = None,
        majority_class: Optional[Any] = None,
        samples_count: int = 0,
        entropy_val: float = 0.0
    ):
        self.attribute = attribute
        self.children = children if children is not None else {}
        self.leaf_value = leaf_value
        self.majority_class = majority_class
        self.samples_count = samples_count
        self.entropy_val = entropy_val
        
    @property
    def is_leaf(self) -> bool:
        return self.leaf_value is not None

class DecisionTreeScratch:
    def __init__(self, max_depth: int = 4, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionTreeNode] = None
        self.target_name: str = ""
        self.default_class: Any = None
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """Treina a árvore de decisão usando LEARN-DECISION-TREE recursivo."""
        df = X.copy()
        target_col = "__target__"
        df[target_col] = np.asarray(y)
        self.target_name = target_col
        self.default_class = df[target_col].mode()[0]
        
        attributes = list(X.columns)
        self.root = self._build_tree(df, attributes, depth=0)
        return self
        
    def _build_tree(self, df: pd.DataFrame, attributes: List[str], depth: int) -> DecisionTreeNode:
        n_samples = len(df)
        y_vals = df[self.target_name].values
        node_entropy = entropy(y_vals)
        majority = df[self.target_name].mode()[0] if n_samples > 0 else self.default_class
        
        # 1. Critério de pureza: se todas as classes forem uniformes
        if len(np.unique(y_vals)) <= 1:
            return DecisionTreeNode(leaf_value=y_vals[0] if n_samples > 0 else self.default_class,
                                    majority_class=majority, samples_count=n_samples, entropy_val=node_entropy)
            
        # 2. Critério de profundidade ou atributos esgotados
        if len(attributes) == 0 or depth >= self.max_depth or n_samples < self.min_samples_split:
            return DecisionTreeNode(leaf_value=majority, majority_class=majority,
                                    samples_count=n_samples, entropy_val=node_entropy)
            
        # 3. Selecionar o atributo com maior Ganho de Informação
        gains = {attr: information_gain(df, attr, self.target_name) for attr in attributes}
        best_attr = max(gains, key=gains.get)
        
        # Se nenhum ganho positivo, vira folha
        if gains[best_attr] <= 1e-6:
            return DecisionTreeNode(leaf_value=majority, majority_class=majority,
                                    samples_count=n_samples, entropy_val=node_entropy)
            
        node = DecisionTreeNode(attribute=best_attr, majority_class=majority,
                                samples_count=n_samples, entropy_val=node_entropy)
        remaining_attrs = [a for a in attributes if a != best_attr]
        
        # 4. Criar nós filhos para cada valor observado do atributo
        for val in df[best_attr].unique():
            subset = df[df[best_attr] == val]
            if len(subset) == 0:
                node.children[val] = DecisionTreeNode(leaf_value=majority, majority_class=majority, samples_count=0)
            else:
                node.children[val] = self._build_tree(subset, remaining_attrs, depth=depth + 1)
                
        return node
        
    def predict_one(self, sample: pd.Series, node: DecisionTreeNode) -> Any:
        """Classifica uma única amostra percorrendo a árvore recursivamente."""
        if node.is_leaf:
            return node.leaf_value
            
        val = sample.get(node.attribute)
        if val in node.children:
            return self.predict_one(sample, node.children[val])
        else:
            # Fallback para classe majoritária se o valor da categoria for não visto
            return node.majority_class
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Classifica um conjunto de amostras."""
        if self.root is None:
            raise ValueError("O modelo precisa ser ajustado com fit() antes de predict().")
        preds = [self.predict_one(row, self.root) for _, row in X.iterrows()]
        return np.array(preds)
        
    def print_tree(self, node: Optional[DecisionTreeNode] = None, indent: str = ""):
        """Exibe a árvore em formato textual identado para interpretabilidade."""
        if node is None:
            node = self.root
        if node.is_leaf:
            print(f"{indent}--> Classe: {node.leaf_value} (amostras: {node.samples_count})")
            return
            
        print(f"{indent}[Atributo: {node.attribute}] (H={node.entropy_val:.3f}, N={node.samples_count})")
        for val, child in node.children.items():
            print(f"{indent}  |-- Se {node.attribute} == '{val}':")
            self.print_tree(child, indent + "      ")
