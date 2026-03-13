import numpy as np
from typing import Dict, Any, List

class EntropyCalculator:
    """
    Calculates Shannon entropy on price direction over the last N ticks.
    H = - sum(p(x) * log(p(x)))
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.directions: List[int] = [] # 1 for up, -1 for down, 0 for neutral
        self.prev_price: Optional[float] = None
        self.counts: Dict[int, int] = {1: 0, -1: 0, 0: 0}

    def update(self, tick: Dict[str, Any]) -> float:
        price = (tick['bid'] + tick['ask']) / 2
        
        if self.prev_price is not None:
            direction = 0
            if price > self.prev_price:
                direction = 1
            elif price < self.prev_price:
                direction = -1
            
            self.directions.append(direction)
            self.counts[direction] += 1
            
            if len(self.directions) > self.window_size:
                removed = self.directions.pop(0)
                self.counts[removed] -= 1
        
        self.prev_price = price
            
        if not self.directions:
            return 0.0
            
        # Compute probabilities from counts
        total = len(self.directions)
        probs = [c / total for c in self.counts.values() if c > 0]
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
