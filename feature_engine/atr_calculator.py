"""
ATR Calculator — Fase 3.
Average True Range para SL/TP dinâmicos.
"""

import logging
from collections import deque
from typing import Dict, Any


class ATRCalculator:
    """
    Calcula ATR (Average True Range) em tempo real a partir de ticks.
    Usado para SL/TP dinâmicos baseados na volatilidade atual.
    
    ATR M2 (janela de 2 minutos):
      < $0.15 → LOW VOL
      $0.15-$0.50 → NORMAL
      $0.50-$1.50 → HIGH VOL
      > $1.50 → PANIC — sistema pausa
    """

    def __init__(self, window_ticks: int = 120):
        self.logger = logging.getLogger("ATRCalculator")
        self.window = window_ticks
        self._highs: deque = deque(maxlen=window_ticks)
        self._lows: deque = deque(maxlen=window_ticks)
        self._closes: deque = deque(maxlen=window_ticks)
        self._prev_close: float = 0.0

    def update(self, tick: Dict[str, Any]) -> float:
        """Atualiza ATR com novo tick. Retorna ATR atual."""
        high  = tick.get('ask', 0)
        low   = tick.get('bid', 0)
        close = (high + low) / 2

        if self._prev_close == 0:
            self._prev_close = close
            return 0.0

        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low  - self._prev_close),
        )

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._prev_close = close

        if len(self._closes) < 10:
            return tr

        import numpy as np
        trs = [
            max(
                self._highs[i] - self._lows[i],
                abs(self._highs[i] - self._closes[i-1]),
                abs(self._lows[i]  - self._closes[i-1]),
            )
            for i in range(1, len(self._closes))
        ]
        atr = float(np.mean(trs))
        return round(atr, 5)

    def get_vol_regime(self, atr: float) -> str:
        """Classifica regime de volatilidade pelo ATR."""
        if atr <= 0:
            return "UNKNOWN"
        elif atr < 0.15:
            return "LOW_VOL"
        elif atr < 0.50:
            return "NORMAL_VOL"
        elif atr < 1.50:
            return "HIGH_VOL"
        else:
            return "PANIC"

    def should_pause(self, atr: float) -> bool:
        """True se ATR acima do limite de segurança."""
        return atr > 1.50
