from typing import Dict, Any, Optional

class OFICalculator:
    """
    Calculates Order Flow Imbalance (OFI) incrementally from tick streams.
    Formula: OFI = (bid_size_current - bid_size_previous) - (ask_size_current - ask_size_previous)
    """
    def __init__(self):
        self.prev_bid_price: Optional[float] = None
        self.prev_bid_size: float = 0
        self.prev_ask_price: Optional[float] = None
        self.prev_ask_size: float = 0

    def update(self, tick: Dict[str, Any]) -> float:
        bid_price = tick['bid']
        # Fallback for volume if L2 size or tick volume is not available/reliable
        bid_size = tick.get('volume', 0.0)
        ask_price = tick['ask']
        ask_size = tick.get('volume', 0.0)
        
        # If no volume, use a price-change proxy (Tick Imbalance)
        if bid_size <= 0:
            if self.prev_bid_price is not None:
                bid_size = 1.0 if bid_price >= self.prev_bid_price else 0.5
            else:
                bid_size = 1.0
        
        if ask_size <= 0:
            if self.prev_ask_price is not None:
                ask_size = 1.0 if ask_price <= self.prev_ask_price else 0.5
            else:
                ask_size = 1.0

        delta_bid_size = 0
        if self.prev_bid_price is not None:
            if bid_price > self.prev_bid_price:
                delta_bid_size = bid_size
            elif bid_price < self.prev_bid_price:
                delta_bid_size = -self.prev_bid_size
            else:
                delta_bid_size = bid_size - self.prev_bid_size

        delta_ask_size = 0
        if self.prev_ask_price is not None:
            if ask_price > self.prev_ask_price:
                delta_ask_size = -self.prev_ask_size
            elif ask_price < self.prev_ask_price:
                delta_ask_size = ask_size
            else:
                delta_ask_size = ask_size - self.prev_ask_size

        self.prev_bid_price = bid_price
        self.prev_bid_size = bid_size
        self.prev_ask_price = ask_price
        self.prev_ask_size = ask_size

        return delta_bid_size - delta_ask_size
