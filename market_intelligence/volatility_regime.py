import logging
import numpy as np

class VolatilityRegime:
    """
    Detects if the market is in HIGH_VOL or LOW_VOL state.
    Uses rolling volatility and comparative thresholds.
    """
    def __init__(self, high_vol_threshold: float = 0.0005):
        self.logger = logging.getLogger("VolatilityRegime")
        self.high_vol_threshold = high_vol_threshold
        self.vol_history = []
        self.window = 100

    def detect(self, current_volatility: float) -> str:
        """
        Detects the volatility regime.
        """
        self.vol_history.append(current_volatility)
        if len(self.vol_history) > self.window:
            self.vol_history.pop(0)

        # Comparative analysis: Is current vol significantly above mean?
        avg_vol = np.mean(self.vol_history) if self.vol_history else 0
        
        if current_volatility > self.high_vol_threshold or (avg_vol > 0 and current_volatility > avg_vol * 1.5):
            return "HIGH_VOL"
            
        return "LOW_VOL"
