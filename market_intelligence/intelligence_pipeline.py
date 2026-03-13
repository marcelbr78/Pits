import logging
from typing import Dict, Any
from .regime_detector import RegimeDetector
from .volatility_regime import VolatilityRegime

class IntelligencePipeline:
    """
    Coordinates market context detection by combining trend and volatility regimes.
    """
    def __init__(self):
        self.logger = logging.getLogger("IntelligencePipeline")
        self.detectors: Dict[str, RegimeDetector] = {}
        self.vol_regimes: Dict[str, VolatilityRegime] = {}

    def get_market_state(self, features: Dict[str, Any]) -> str:
        """
        Calculates the composite market state for a symbol.
        """
        symbol = features['symbol']
        mid = (features.get('bid', 0) + features.get('ask', 0)) / 2
        
        # Fallback if features doesn't have bid/ask (get from tick if possible)
        if mid == 0 and 'vwap' in features:
            mid = features['vwap']
        vwap = features.get('vwap', 0)
        vol = features.get('volatility', 0)
        
        # Approximate vwap_std if not stored (Phase 1 simplicity)
        # In a full engine, this would come from FeatureEngine
        vwap_std = abs(mid - vwap) * 0.5 # Placeholder approximation
        
        if symbol not in self.detectors:
            self.detectors[symbol] = RegimeDetector()
            self.vol_regimes[symbol] = VolatilityRegime()

        regime = self.detectors[symbol].detect(mid, vwap, vwap_std)
        vol_state = self.vol_regimes[symbol].detect(vol)
        
        market_state = f"{regime}_{vol_state}"
        
        # Log transition only when it changes (optional but helpful)
        self.logger.debug(f"[{symbol}] Market State: {market_state}")
        
        return market_state
