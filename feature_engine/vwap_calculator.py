from typing import Dict, Any

class VWAPCalculator:
    """
    Calculates Volume Weighted Average Price (VWAP) incrementally.
    VWAP = sum(price * volume) / sum(volume)
    """
    def __init__(self):
        self.cumulative_pv: float = 0
        self.cumulative_volume: float = 0
        self.last_date: Optional[str] = None

    def update(self, tick: Dict[str, Any]) -> float:
        # Session Reset logic based on date
        timestamp = tick.get('timestamp')
        if timestamp:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            current_date = dt.strftime('%Y-%m-%d')
            
            if self.last_date and current_date != self.last_date:
                self.reset()
            self.last_date = current_date

        # Use average of bid/ask if 'last' is not available
        price = tick.get('last', (tick['bid'] + tick['ask']) / 2)
        volume = tick['volume']

        self.cumulative_pv += price * volume
        self.cumulative_volume += volume

        if self.cumulative_volume == 0:
            return price
            
        return self.cumulative_pv / self.cumulative_volume

    def reset(self):
        """Resets session VWAP."""
        self.cumulative_pv = 0
        self.cumulative_volume = 0
