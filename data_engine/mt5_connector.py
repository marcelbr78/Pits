import MetaTrader5 as mt5
import logging
from typing import Optional, Dict, Any

class MT5Connector:
    """
    Handles connection and basic data retrieval from MetaTrader 5.
    """
    def __init__(self):
        self.logger = logging.getLogger("MT5Connector")
        self.connected = False

    def connect(self) -> bool:
        """Initializes connection to MT5."""
        if not mt5.initialize():
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        self.connected = True
        self.logger.info("MetaTrader 5 connected successfully.")
        return True

    def shutdown(self):
        """Closes MT5 connection."""
        mt5.shutdown()
        self.connected = False
        self.logger.info("MetaTrader 5 connection closed.")

    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves the last tick for a given symbol."""
        if not self.connected:
            return None
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            err = mt5.last_error()
            self.logger.warning(f"Failed to get tick for {symbol}: {err}")
            return None
            
        return {
            "symbol": symbol,
            "timestamp": tick.time,
            "time_msc": tick.time_msc,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume
        }

    def get_orderbook(self, symbol: str) -> Optional[tuple]:
        """Retrieves current order book (DOM) for a symbol."""
        if not self.connected:
            return None
            
        book = mt5.market_book_get(symbol)
        if book is None:
            self.logger.warning(f"Failed to get orderbook for {symbol}: {mt5.last_error()}")
            return None
            
        return book

    def is_connected(self) -> bool:
        return self.connected
