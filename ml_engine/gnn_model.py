"""
Graph Neural Network Real — Fase 5.

Implementação real da GNN com PyTorch Geometric.
Cada ativo é um nó. As arestas são correlações dinâmicas.
Message passing propaga informação entre ativos.

Arquitetura:
  Nodes = 6 ativos (WTI, Brent, Gold, SP500, DXY, VIX)
  Edges = correlações com pesos dinâmicos
  GCN   = 2 camadas de Graph Convolution
  Output = probabilidade de movimento para cada nó

Se PyTorch Geometric não disponível, fallback para
correlação linear simples.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional


# Mapeamento símbolo → índice do nó
NODE_INDEX = {
    'USOILm':  0,
    'UKOILm':  1,
    'XAUUSDm': 2,
    'BTCUSDm': 3,
    'ETHUSDm': 4,
    'VIX':     5,
}

# Arestas base (pares correlacionados)
BASE_EDGES = [
    (0, 1), (1, 0),  # WTI ↔ Brent
    (0, 2), (2, 0),  # WTI ↔ Gold
    (0, 3), (3, 0),  # WTI ↔ BTC
    (1, 2), (2, 1),  # Brent ↔ Gold
    (2, 3), (3, 2),  # Gold ↔ BTC
    (0, 5), (5, 0),  # WTI ↔ VIX
]


class TradingGNN:
    """
    GNN para trading multi-ativo.
    
    Em modo completo (PyG disponível):
      - GCNConv 2 camadas
      - Pesos de aresta dinâmicos por regime
    
    Em modo fallback (sem PyG):
      - Média ponderada por correlação
    """

    def __init__(
        self,
        n_nodes: int = 6,
        n_features_per_node: int = 8,
        hidden_dim: int = 32,
        model_path: str = "models/gnn_model.pt",
    ):
        self.logger = logging.getLogger("TradingGNN")
        self.n_nodes    = n_nodes
        self.n_features = n_features_per_node
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        self.is_trained = False

        # Estado atual de cada nó
        self._node_features: Dict[str, List[float]] = {}

        self._model = None
        self._pyg_available = False

        self._init_pyg()

    def _init_pyg(self):
        try:
            import torch
            from torch_geometric.nn import GCNConv
            import torch.nn as nn

            class _GNN(nn.Module):
                def __init__(self, n_feat, hidden):
                    super().__init__()
                    self.conv1 = GCNConv(n_feat, hidden)
                    self.conv2 = GCNConv(hidden, hidden // 2)
                    self.fc    = nn.Linear(hidden // 2, 1)
                    self.relu  = nn.ReLU()
                    self.sig   = nn.Sigmoid()

                def forward(self, x, edge_index, edge_weight=None):
                    x = self.relu(self.conv1(x, edge_index, edge_weight))
                    x = self.relu(self.conv2(x, edge_index, edge_weight))
                    return self.sig(self.fc(x)).squeeze(-1)

            self._torch = torch
            self._GNN   = _GNN
            self._model = _GNN(self.n_features, self.hidden_dim)
            self._pyg_available = True
            self.load_model()
            self.logger.info("GNN (PyTorch Geometric) inicializada.")

        except ImportError:
            self.logger.warning(
                "PyTorch Geometric não disponível. "
                "GNN usa fallback de correlação linear. "
                "Para instalar: pip install torch-geometric"
            )

    def update_node(self, symbol: str, features: List[float]):
        """Atualiza features de um nó (ativo)."""
        vec = features[:self.n_features]
        while len(vec) < self.n_features:
            vec.append(0.0)
        self._node_features[symbol] = vec

    def predict(self, target_symbol: str = 'USOILm', macro_regime: str = 'NEUTRAL') -> float:
        """
        Prediz probabilidade de movimento UP para o símbolo alvo.
        Usa informação de todos os nós conectados.
        """
        if len(self._node_features) < 2:
            return 0.5

        if self._pyg_available and self.is_trained:
            return self._predict_gnn(target_symbol, macro_regime)
        else:
            return self._predict_fallback(target_symbol, macro_regime)

    def _predict_gnn(self, target: str, macro: str) -> float:
        try:
            import torch

            # Monta matriz de features dos nós
            nodes = list(NODE_INDEX.keys())
            X = []
            for node in nodes:
                feat = self._node_features.get(node, [0.0] * self.n_features)
                X.append(feat)

            x = torch.FloatTensor(X)

            # Arestas com pesos dinâmicos por regime
            edges_src = [e[0] for e in BASE_EDGES]
            edges_dst = [e[1] for e in BASE_EDGES]
            edge_index = torch.LongTensor([edges_src, edges_dst])
            edge_weight = self._get_edge_weights(macro)

            self._model.eval()
            with torch.no_grad():
                probs = self._model(x, edge_index, edge_weight)

            node_idx = NODE_INDEX.get(target, 0)
            return round(float(probs[node_idx].item()), 4)

        except Exception as e:
            self.logger.error(f"GNN predict erro: {e}")
            return 0.5

    def _predict_fallback(self, target: str, macro: str) -> float:
        """Fallback sem PyG — média ponderada por correlação."""
        from risk_engine.portfolio_risk import CORRELATIONS, WAR_CORRELATIONS

        corr_map = WAR_CORRELATIONS if macro == 'WAR' else CORRELATIONS
        target_corrs = corr_map.get(target, {})

        target_feat = self._node_features.get(target)
        if not target_feat:
            return 0.5

        # Probabilidade base do ativo alvo (primeira feature = prob_up)
        base_prob = target_feat[0] if target_feat else 0.5

        # Ajuste ponderado pelos outros ativos
        weighted_sum = base_prob
        total_weight = 1.0

        for sym, feat in self._node_features.items():
            if sym == target or not feat:
                continue
            corr = abs(target_corrs.get(sym, 0.3))
            other_prob = feat[0] if feat else 0.5
            weighted_sum += corr * other_prob
            total_weight += corr

        return round(weighted_sum / total_weight, 4)

    def _get_edge_weights(self, macro: str):
        """Pesos dinâmicos das arestas por regime."""
        import torch
        from risk_engine.portfolio_risk import CORRELATIONS, WAR_CORRELATIONS

        corr_map = WAR_CORRELATIONS if macro == 'WAR' else CORRELATIONS
        nodes = list(NODE_INDEX.keys())
        weights = []

        for src, dst in BASE_EDGES:
            sym_src = nodes[src]
            sym_dst = nodes[dst]
            w = corr_map.get(sym_src, {}).get(sym_dst, 0.3)
            weights.append(w)

        return torch.FloatTensor(weights)

    def train(self, node_sequences: np.ndarray, labels: np.ndarray, epochs: int = 50):
        """Treina GNN com sequências históricas multi-ativo."""
        if not self._pyg_available:
            self.logger.warning("PyG não disponível — treino GNN pulado.")
            return

        try:
            import torch
            from torch_geometric.data import Data, DataLoader as GDataLoader

            self.logger.info(f"Treinando GNN: {len(node_sequences)} amostras × {epochs} epochs")
            optimizer = self._torch.optim.Adam(self._model.parameters(), lr=0.001)
            criterion = self._torch.nn.BCELoss()

            edges_src = [e[0] for e in BASE_EDGES]
            edges_dst = [e[1] for e in BASE_EDGES]
            edge_index = torch.LongTensor([edges_src, edges_dst])

            self._model.train()
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(node_sequences), 32):
                    batch_X = torch.FloatTensor(node_sequences[i:i+32])
                    batch_y = torch.FloatTensor(labels[i:i+32])

                    optimizer.zero_grad()
                    for j in range(len(batch_X)):
                        x = batch_X[j]
                        pred = self._model(x, edge_index)
                        loss = criterion(pred[0], batch_y[j])
                        loss.backward()
                        total_loss += loss.item()

                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"GNN Epoch {epoch+1}/{epochs} — loss: {total_loss:.4f}")

            self.is_trained = True
            self.save_model()

        except Exception as e:
            self.logger.error(f"Erro ao treinar GNN: {e}")

    def save_model(self):
        if self._model is None or not self.is_trained:
            return
        try:
            import torch, os
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self._model.state_dict(), self.model_path)
        except Exception as e:
            self.logger.error(f"Erro ao salvar GNN: {e}")

    def load_model(self):
        if self._model is None:
            return
        try:
            import torch, os
            if os.path.exists(self.model_path):
                state = torch.load(self.model_path, map_location='cpu')
                self._model.load_state_dict(state)
                self.is_trained = True
                self.logger.info("GNN carregada.")
        except Exception as e:
            self.logger.warning(f"GNN não carregada: {e}")
