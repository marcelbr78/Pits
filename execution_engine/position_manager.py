import MetaTrader5 as mt5
import logging
from typing import Dict, Any, List, Optional

class PositionManager:
    """
    Tracks open positions and prevents duplicate trades.
    """
    def __init__(self, magic_number: int = 123456):
        self.logger = logging.getLogger("PositionManager")
        self.magic_number = magic_number

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Wraps mt5.positions_get to filter by magic number."""
        positions = mt5.positions_get(magic=self.magic_number)
        if positions is None or len(positions) == 0:
            return []
            
        pos_list = []
        for p in positions:
            # Map namedtuple to dict for easier handling
            pos_dict = p._asdict()
            if symbol and pos_dict['symbol'] != symbol:
                continue
            pos_list.append(pos_dict)
            
        return pos_list

    def has_position(self, symbol: str) -> bool:
        """Returns True if a position is already open for the symbol."""
        positions = self.get_open_positions(symbol)
        return len(positions) > 0
