from typing import Dict, Any, List
import logging
from .ofi_calculator import OFICalculator
from .vwap_calculator import VWAPCalculator
from .volatility_calculator import VolatilityCalculator
from .entropy_calculator import EntropyCalculator

class FeaturePipeline:
    """
    Orchestrates multiple Feature Calculators for each symbol.
    """
    def __init__(self, symbols: List[str]):
        self.logger = logging.getLogger("FeaturePipeline")
        self.symbols = symbols
        self.calculators: Dict[str, Dict[str, Any]] = {}
        
        for symbol in symbols:
            self.calculators[symbol] = {
                'ofi': OFICalculator(),
                'vwap': VWAPCalculator(),
                'volatility': VolatilityCalculator(window_size=100),
                'entropy': EntropyCalculator(window_size=50)
            }

    def process_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates all features for a given tick."""
        symbol = tick['symbol']
        if symbol not in self.calculators:
            return tick

        calcs = self.calculators[symbol]
        
        # Standardized feature vector output
        features = {
            'symbol': symbol,
            'timestamp': tick.get('timestamp'),
            'ofi': round(calcs['ofi'].update(tick), 4),
            'vwap': round(calcs['vwap'].update(tick), 4),
            'spread': round(tick.get('spread', tick['ask'] - tick['bid']), 6),
            'volatility': round(calcs['volatility'].update(tick), 8),
            'entropy': round(calcs['entropy'].update(tick), 6)
        }
        
        return features

    def reset_sessions(self):
        """Manually triggers session reset for all calculators."""
        for symbol in self.symbols:
            self.calculators[symbol]['vwap'].reset()
        self.logger.info("Session features reset for all symbols.")
