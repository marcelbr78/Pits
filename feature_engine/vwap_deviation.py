import logging
from typing import Dict, Any

class VWAPDeviationCalculator:
    """
    VWAP Deviation — Fase 2.
    
    Calcula desvio do preço em relação ao VWAP da sessão.
    Institucional usa isso como referência de preço justo.
    
    Desvio negativo (preço < VWAP) + OBI buy alto = 
      compra institucional em andamento = sinal forte.
    """

    def __init__(self):
        self.logger = logging.getLogger("VWAPDeviationCalculator")
        self._sum_pv = 0.0
        self._sum_v  = 0.0
        self._prices = []
        self._current_date = None

    def update(self, tick: Dict[str, Any]) -> Dict[str, float]:
        """
        Atualiza VWAP e retorna métricas de desvio.
        
        Returns dict com:
          vwap          : VWAP da sessão
          deviation_pct : desvio percentual (preço - VWAP) / VWAP
          deviation_atr : desvio em múltiplos de ATR (se atr fornecido)
          z_score       : z-score do desvio
        """
        import datetime
        import numpy as np

        ts = tick.get('timestamp', 0)
        date_now = datetime.datetime.fromtimestamp(ts).date() if ts else None

        # Reset diário
        if date_now and date_now != self._current_date:
            self._sum_pv = 0.0
            self._sum_v  = 0.0
            self._prices = []
            self._current_date = date_now

        mid   = (tick.get('bid', 0) + tick.get('ask', 0)) / 2
        vol   = tick.get('volume', 1.0) or 1.0

        if mid == 0:
            return {'vwap': 0.0, 'deviation_pct': 0.0, 'z_score': 0.0}

        self._sum_pv += mid * vol
        self._sum_v  += vol
        self._prices.append(mid)

        vwap = self._sum_pv / self._sum_v if self._sum_v > 0 else mid

        deviation_pct = (mid - vwap) / vwap if vwap > 0 else 0.0

        # Z-score usando últimos 200 preços
        recent = self._prices[-200:]
        if len(recent) >= 20:
            std = np.std(recent)
            mean = np.mean(recent)
            z_score = (mid - mean) / std if std > 0 else 0.0
        else:
            z_score = 0.0

        return {
            'vwap':          round(vwap, 5),
            'deviation_pct': round(deviation_pct, 6),
            'z_score':       round(z_score, 4),
        }
