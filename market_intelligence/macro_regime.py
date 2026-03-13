import logging
import numpy as np
from collections import deque
from typing import Dict, Any

class MacroRegimeDetector:
    """
    Regime Macro — Fase 2 (3ª camada do regime detection).
    
    Detecta o contexto global do mercado usando correlações
    entre ativos. Muda lentamente — dias a semanas.
    
    Regimes:
      RISK_ON   : SP500 subindo, VIX < 20, DXY fraco
      RISK_OFF  : SP500 caindo, VIX > 25, DXY forte
      WAR       : VIX > 30, Ouro subindo forte, notícias militares
      INFLATION : CPI alto implícito (Ouro + Petróleo sobem juntos)
      CRISIS    : Tudo cai — pandemia, recessão
    """

    def __init__(self, window: int = 200):
        self.logger = logging.getLogger("MacroRegimeDetector")
        self.window = window
        # Histórico de preços por ativo
        self._prices: Dict[str, deque] = {}
        self._current_regime = "UNKNOWN"

    def update_asset(self, symbol: str, mid_price: float):
        """Atualiza histórico de um ativo."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.window)
        self._prices[symbol].append(mid_price)

    def detect(self) -> str:
        """
        Detecta regime macro atual baseado nos ativos disponíveis.
        Retorna string do regime.
        """
        ret = self._get_recent_returns()

        vix_ret    = ret.get('VIX', 0)
        gold_ret   = ret.get('XAUUSDm', 0)
        sp500_ret  = ret.get('BTCUSDm', 0)   # Proxy se SP500 não disponível
        oil_ret    = ret.get('USOILm', ret.get('UKOILm', 0))

        # WAR: VIX subindo + Ouro subindo + Petróleo subindo fortemente
        if vix_ret > 0.02 and gold_ret > 0.005 and oil_ret > 0.01:
            regime = "WAR"

        # CRISIS: tudo caindo
        elif oil_ret < -0.03 and gold_ret < -0.01 and sp500_ret < -0.02:
            regime = "CRISIS"

        # RISK_OFF: VIX alto, Ouro subindo, petróleo instável
        elif vix_ret > 0.015 and gold_ret > 0.003:
            regime = "RISK_OFF"

        # INFLATION: Ouro + Petróleo subindo juntos moderadamente
        elif gold_ret > 0.002 and oil_ret > 0.005:
            regime = "INFLATION"

        # RISK_ON: mercado calmo, petróleo subindo moderado
        elif vix_ret < 0 and oil_ret > 0:
            regime = "RISK_ON"

        else:
            regime = "NEUTRAL"

        if regime != self._current_regime:
            self.logger.info(f"Regime macro mudou: {self._current_regime} → {regime}")
            self._current_regime = regime

        return regime

    def _get_recent_returns(self, lookback: int = 20) -> Dict[str, float]:
        """Retornos recentes de cada ativo."""
        returns = {}
        for symbol, prices in self._prices.items():
            arr = list(prices)
            if len(arr) >= lookback + 1:
                past  = arr[-(lookback + 1)]
                now   = arr[-1]
                if past > 0:
                    returns[symbol] = (now - past) / past
        return returns
