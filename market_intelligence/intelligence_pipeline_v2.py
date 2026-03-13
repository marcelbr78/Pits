import logging
from typing import Dict, Any, Optional
from .regime_detector import RegimeDetector
from .volatility_regime import VolatilityRegime
from .macro_regime import MacroRegimeDetector
from .economic_calendar import EconomicCalendar

class MarketStateV2:
    """Estado completo do mercado — 3 camadas + calendário."""
    def __init__(self, macro, volatility_regime, trend, calendar_modifier):
        self.macro = macro
        self.volatility_regime = volatility_regime
        self.trend = trend
        self.calendar_modifier = calendar_modifier
        # Combined state legível
        self.combined = f"{macro}_{volatility_regime}_{trend}"

class IntelligencePipelineV2:
    """
    Intelligence Pipeline Fase 2 — 3 camadas de regime.
    
    Camada 1 — Macro: WAR / RISK_ON / RISK_OFF / INFLATION / CRISIS
    Camada 2 — Volatilidade: HIGH_VOL / NORMAL_VOL / LOW_VOL
    Camada 3 — Micro: TRENDING / RANGING
    
    + Calendário econômico integrado.
    
    Compatível com IntelligencePipeline Fase 1 — substitui diretamente.
    """

    def __init__(self):
        self.logger = logging.getLogger("IntelligencePipelineV2")
        self.detectors: Dict[str, RegimeDetector] = {}
        self.vol_regimes: Dict[str, VolatilityRegime] = {}
        self.macro_detector = MacroRegimeDetector(window=200)
        self.calendar = EconomicCalendar()

    def get_market_state(self, features: Dict[str, Any]) -> MarketStateV2:
        """
        Calcula estado completo do mercado para um símbolo.
        API compatível com IntelligencePipeline Fase 1.
        """
        symbol  = features['symbol']
        mid     = (features.get('bid', 0) + features.get('ask', 0)) / 2
        if mid == 0:
            mid = features.get('vwap', 0)
        vwap    = features.get('vwap', 0)
        vol     = features.get('volatility', 0)

        # Atualiza macro com preço atual deste ativo
        self.macro_detector.update_asset(symbol, mid)

        # Camada 1 — Regime macro
        macro = self.macro_detector.detect()

        # Camada 2 — Regime de volatilidade (com 4 estados vs 2 da Fase 1)
        if symbol not in self.vol_regimes:
            self.vol_regimes[symbol] = VolatilityRegime()
        vol_state_raw = self.vol_regimes[symbol].detect(vol)
        vol_state = self._classify_vol(vol, vol_state_raw)

        # Camada 3 — Regime micro (TRENDING/RANGING)
        vwap_std = abs(mid - vwap) * 0.5
        if symbol not in self.detectors:
            self.detectors[symbol] = RegimeDetector()
        micro = self.detectors[symbol].detect(mid, vwap, vwap_std)

        # Calendário econômico
        calendar_modifier = self.calendar.get_trading_modifier()

        state = MarketStateV2(
            macro=macro,
            volatility_regime=vol_state,
            trend=micro,
            calendar_modifier=calendar_modifier
        )

        self.logger.debug(
            f"[{symbol}] {state.combined} | "
            f"Calendar: {calendar_modifier['reason']}"
        )

        return state

    def get_next_event(self) -> Dict[str, Any]:
        """Retorna próximo evento do calendário."""
        return self.calendar.get_next_event()

    def _classify_vol(self, current_vol: float, raw_state: str) -> str:
        """Expande 2 estados para 4 estados de volatilidade."""
        if current_vol == 0:
            return "UNKNOWN"
        # Thresholds para WTI (ajustar por backtest)
        if current_vol > 0.002:
            return "PANIC"
        elif current_vol > 0.0008:
            return "HIGH_VOL"
        elif raw_state == "HIGH_VOL":
            return "NORMAL_VOL"
        else:
            return "LOW_VOL"
