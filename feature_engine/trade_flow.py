import logging
from collections import deque
from typing import Dict, Any

class TradeFlowAnalyzer:
    """
    Trade Flow Toxicity (TFT) — Fase 2.
    Detecta quem está sendo agressivo: compradores ou vendedores.
    
    Usa tick flags do MT5:
      TICK_FLAG_BUY  = comprador bateu no ask (buyer-initiated)
      TICK_FLAG_SELL = vendedor bateu no bid (seller-initiated)
    
    TFT = buyer_ticks / total_ticks (janela deslizante)
    TFT > 0.70 → compradores dominantes → sinal buy forte
    TFT < 0.30 → vendedores dominantes → não opera
    """

    # MT5 tick flags
    TICK_FLAG_BUY  = 2
    TICK_FLAG_SELL = 4

    def __init__(self, window_seconds: int = 60):
        self.logger = logging.getLogger("TradeFlowAnalyzer")
        self.window_seconds = window_seconds
        # deque de (timestamp_ms, is_buy)
        self._ticks: deque = deque()

    def update(self, tick: Dict[str, Any]) -> float:
        """
        Processa um tick e retorna TFT atual (0.0 a 1.0).
        0.5 = neutro, >0.7 = buy dominante, <0.3 = sell dominante.
        """
        ts_ms = tick.get('time_msc', tick.get('timestamp', 0) * 1000)
        flags = tick.get('flags', 0)

        is_buy: bool
        if flags & self.TICK_FLAG_BUY:
            is_buy = True
        elif flags & self.TICK_FLAG_SELL:
            is_buy = False
        else:
            # Sem flag — inferir pela direção do preço
            is_buy = tick.get('bid', 0) >= tick.get('ask', 0) * 0.9999

        self._ticks.append((ts_ms, is_buy))

        # Remove ticks fora da janela
        cutoff = ts_ms - (self.window_seconds * 1000)
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()

        if not self._ticks:
            return 0.5

        total = len(self._ticks)
        buyers = sum(1 for _, b in self._ticks if b)
        tft = buyers / total
        return round(tft, 4)

    def get_signal(self, tft: float) -> str:
        """Converte TFT em sinal interpretável."""
        if tft > 0.70:
            return "BUY_DOMINANT"
        elif tft < 0.30:
            return "SELL_DOMINANT"
        else:
            return "NEUTRAL"
