import logging
from collections import deque
from typing import Dict, Any, Optional
import time

class LagFeatureEngine:
    """
    Cross-Asset Lag Features — Fase 2.
    
    Ensina ao XGBoost a cadeia de causa e efeito entre ativos:
      Gold move t-10s  → WTI move t=0
      DXY  move t-5s   → WTI move t=0
      VIX  spike t-15s → WTI move t=0
    
    Armazena histórico de retornos por ativo e
    fornece vetor de lag features para qualquer momento.
    """

    LAG_WINDOWS = {
        'XAUUSDm': [5, 10, 30, 60],   # Ouro precede WTI em 10-60s
        'DXY':     [3, 5, 10, 20],     # DXY precede WTI em 5-20s  (correlação inversa)
        'VIX':     [10, 15, 30],       # VIX spike precede WTI em 15-30s
        'BTCUSDm': [5, 10],            # BTC correlação baixa mas rápida
        'UKOILm':  [2, 5],             # Brent precede WTI em 2-5s
    }

    def __init__(self, max_history_seconds: int = 120):
        self.logger = logging.getLogger("LagFeatureEngine")
        self.max_history_ms = max_history_seconds * 1000
        # {symbol: deque de (ts_ms, mid_price)}
        self._price_history: Dict[str, deque] = {}

    def update(self, symbol: str, tick: Dict[str, Any]):
        """Registra preço atual do ativo no histórico."""
        ts_ms = tick.get('time_msc', int(time.time() * 1000))
        mid = (tick.get('bid', 0) + tick.get('ask', 0)) / 2
        if mid == 0:
            return

        if symbol not in self._price_history:
            self._price_history[symbol] = deque()

        self._price_history[symbol].append((ts_ms, mid))

        # Limpar histórico antigo
        cutoff = ts_ms - self.max_history_ms
        while self._price_history[symbol] and self._price_history[symbol][0][0] < cutoff:
            self._price_history[symbol].popleft()

    def get_lag_features(self, current_ts_ms: Optional[int] = None) -> Dict[str, float]:
        """
        Retorna dicionário com retornos passados de cada ativo nas janelas definidas.
        Ex: {'XAUUSDm_ret_10s': 0.0023, 'DXY_ret_5s': -0.0011, ...}
        """
        if current_ts_ms is None:
            current_ts_ms = int(time.time() * 1000)

        features: Dict[str, float] = {}

        for symbol, lag_windows in self.LAG_WINDOWS.items():
            if symbol not in self._price_history or not self._price_history[symbol]:
                for lag_s in lag_windows:
                    features[f"{symbol}_ret_{lag_s}s"] = 0.0
                continue

            history = self._price_history[symbol]
            current_price = history[-1][1]

            for lag_s in lag_windows:
                lag_ms = lag_s * 1000
                target_ts = current_ts_ms - lag_ms

                # Busca preço mais próximo do timestamp alvo
                past_price = self._find_price_at(history, target_ts)

                if past_price and past_price > 0:
                    ret = (current_price - past_price) / past_price
                    features[f"{symbol}_ret_{lag_s}s"] = round(ret, 6)
                else:
                    features[f"{symbol}_ret_{lag_s}s"] = 0.0

        return features

    def _find_price_at(self, history: deque, target_ts_ms: int) -> Optional[float]:
        """Busca preço mais próximo de um timestamp no histórico."""
        best_price = None
        best_diff = float('inf')

        for ts, price in history:
            diff = abs(ts - target_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_price = price
            elif diff > best_diff:
                break  # histórico é ordenado, pode parar

        # Só retorna se o preço está dentro de 5s do alvo
        if best_diff < 5000:
            return best_price
        return None
