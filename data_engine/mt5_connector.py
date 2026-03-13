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

    def get_historical_ticks(self, symbol: str, count: int) -> Optional[List[Dict[str, Any]]]:
        """Retrieves historical ticks from MT5."""
        if not self.connected:
            return None
            
        import datetime
        import pandas as pd
        
        # Try fetching by count first
        start_time = datetime.datetime.now()
        ticks = mt5.copy_ticks_from(symbol, start_time, count, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            # Fallback: try last 24 hours
            end = datetime.datetime.now()
            start = end - datetime.timedelta(hours=24)
            ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            err = mt5.last_error()
            self.logger.warning(f"Failed to get historical ticks for {symbol}: {err}")
            return None
            
        # Convert structured numpy array to list of dicts using field names
        tick_list = []
        for t in ticks:
            # Struct fields: time, bid, ask, last, volume, time_msc, flags, volume_real
            tick_list.append({
                "symbol": symbol,
                "timestamp": int(t['time']),
                "time_msc": int(t['time_msc']),
                "bid": float(t['bid']),
                "ask": float(t['ask']),
                "last": float(t['last']),
                "volume": float(t['volume_real'] if 'volume_real' in t.dtype.names else t['volume'])
            })
        return tick_list

    def is_connected(self) -> bool:
        return self.connected
