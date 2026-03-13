import logging
import numpy as np
from typing import Dict, Any

class RegimeDetector:
    """
    Detects if the market is TRENDING or RANGING.
    Uses VWAP deviation and recent price movement consistency.
    """
    def __init__(self, deviation_threshold: float = 2.0, window: int = 50):
        self.logger = logging.getLogger("RegimeDetector")
        self.deviation_threshold = deviation_threshold # Multiplier for standard deviation
        self.window = window
        self.history = []

    def detect(self, mid_price: float, vwap: float, vwap_std: float = 0.0) -> str:
        """
        Detects the current trend regime.
        """
        if vwap == 0 or vwap_std == 0:
            return "RANGING"

        # Z-Score of price relative to VWAP
        z_score = abs(mid_price - vwap) / vwap_std if vwap_std > 0 else 0
        
        # Track history for consistency check
        self.history.append(mid_price)
        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < self.window:
            return "RANGING"

        # Check for consistent direction (Trending)
        # Using a simple linear fit slope or just start vs end
        total_move = abs(self.history[-1] - self.history[0])
        avg_move = np.mean(np.abs(np.diff(self.history)))
        
        # Efficiency Ratio (ER)
        # 1.0 = Perfect trend, 0.0 = noise
        er = total_move / (np.sum(np.abs(np.diff(self.history))) + 1e-9)

        if z_score > self.deviation_threshold or er > 0.4:
            return "TRENDING"
        
        return "RANGING"
