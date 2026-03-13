"""
LSTM Model — Fase 5.

Long Short-Term Memory para capturar padrões temporais longos.
Aprende sequências de horas que se repetem:
  - Padrão de abertura NY Spike (13:30-15:00 UTC)
  - Padrão pré-EIA (acumulação gradual)
  - Padrão pós-OPEP (continuação de tendência)

Usa PyTorch. Se não disponível, retorna 0.5 (neutro).
"""

import logging
import numpy as np
from typing import List, Optional
from collections import deque


class LSTMModel:
    """
    LSTM para séries temporais de features de trading.
    
    Arquitetura:
      Input:  sequência de T timesteps × N features
      LSTM:   2 camadas × 64 hidden units
      Output: probabilidade de movimento UP (0-1)
    """

    def __init__(
        self,
        n_features: int = 25,
        seq_length: int = 60,
        hidden_size: int = 64,
        n_layers: int = 2,
        model_path: str = "models/lstm_model.pt",
    ):
        self.logger = logging.getLogger("LSTMModel")
        self.n_features  = n_features
        self.seq_length  = seq_length
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.model_path  = model_path
        self.is_trained  = False

        self._buffer: deque = deque(maxlen=seq_length)
        self._model = None
        self._device = None

        self._init_torch()

    def _init_torch(self):
        """Inicializa PyTorch se disponível."""
        try:
            import torch
            import torch.nn as nn

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            class _LSTM(nn.Module):
                def __init__(self, n_feat, hidden, layers):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=n_feat,
                        hidden_size=hidden,
                        num_layers=layers,
                        batch_first=True,
                        dropout=0.2 if layers > 1 else 0,
                    )
                    self.bn   = nn.BatchNorm1d(hidden)
                    self.fc   = nn.Linear(hidden, 1)
                    self.sig  = nn.Sigmoid()

                def forward(self, x):
                    out, _ = self.lstm(x)
                    last   = out[:, -1, :]
                    last   = self.bn(last)
                    return self.sig(self.fc(last)).squeeze(-1)

            self._model = _LSTM(self.n_features, self.hidden_size, self.n_layers)
            self._model.to(self._device)
            self._torch = torch
            self._nn    = nn

            self.load_model()
            self.logger.info(f"LSTM inicializado no {self._device}")

        except ImportError:
            self.logger.warning("PyTorch não disponível — LSTM desativado. Retorna 0.5.")
            self._model = None

    def update_buffer(self, feature_vector: List[float]):
        """Adiciona vetor de features atual ao buffer de sequência."""
        # Garante tamanho fixo
        vec = feature_vector[:self.n_features]
        while len(vec) < self.n_features:
            vec.append(0.0)
        self._buffer.append(vec)

    def predict(self) -> float:
        """
        Prediz probabilidade UP baseado na sequência atual.
        Requer seq_length amostras no buffer.
        """
        if self._model is None:
            return 0.5

        if len(self._buffer) < self.seq_length:
            return 0.5

        if not self.is_trained:
            return 0.5

        try:
            import torch
            seq = np.array(list(self._buffer), dtype=np.float32)
            seq = np.nan_to_num(seq, 0)
            x   = torch.FloatTensor(seq).unsqueeze(0).to(self._device)

            self._model.eval()
            with torch.no_grad():
                prob = float(self._model(x).cpu().item())

            return round(prob, 4)

        except Exception as e:
            self.logger.error(f"Erro LSTM predict: {e}")
            return 0.5

    def train(self, sequences: np.ndarray, labels: np.ndarray, epochs: int = 30):
        """
        Treina o LSTM com sequências históricas.
        
        Args:
            sequences: (N, seq_length, n_features)
            labels:    (N,) — 0 ou 1
        """
        if self._model is None:
            return

        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            X = torch.FloatTensor(sequences).to(self._device)
            y = torch.FloatTensor(labels).to(self._device)

            dataset = TensorDataset(X, y)
            loader  = DataLoader(dataset, batch_size=64, shuffle=True)

            optimizer = self._torch.optim.Adam(self._model.parameters(), lr=0.001)
            criterion = self._nn.BCELoss()

            self._model.train()
            for epoch in range(epochs):
                total_loss = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = self._model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

            self.is_trained = True
            self.save_model()

        except Exception as e:
            self.logger.error(f"Erro ao treinar LSTM: {e}")

    def save_model(self):
        if self._model is None or not self.is_trained:
            return
        try:
            import torch, os
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self._model.state_dict(), self.model_path)
            self.logger.info(f"LSTM salvo em {self.model_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar LSTM: {e}")

    def load_model(self):
        if self._model is None:
            return
        try:
            import torch, os
            if os.path.exists(self.model_path):
                state = torch.load(self.model_path, map_location=self._device)
                self._model.load_state_dict(state)
                self.is_trained = True
                self.logger.info(f"LSTM carregado de {self.model_path}")
        except Exception as e:
            self.logger.warning(f"LSTM não carregado: {e}")
