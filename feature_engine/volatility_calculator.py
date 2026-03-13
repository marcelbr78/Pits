import numpy as np
from typing import Dict, Any, List

class VolatilityCalculator:
    """
    Calculates rolling volatility over the last N ticks.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prices: List[float] = []

    def update(self, tick: Dict[str, Any]) -> float:
        price = (tick['bid'] + tick['ask']) / 2
        self.prices.append(price)
        
        if len(self.prices) > self.window_size:
            self.prices.pop(0)
            
        if len(self.prices) < 2:
            return 0.0
            
        # Calculate log returns
        prices_arr = np.array(self.prices)
        returns = np.diff(np.log(prices_arr))
        
        return np.std(returns)
