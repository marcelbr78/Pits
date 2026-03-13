import time
import logging
from typing import List, Dict, Any, Callable
from .mt5_connector import MT5Connector

class TickCollector:
    """
    Collects real-time tick data for multiple symbols.
    """
    def __init__(self, connector: MT5Connector, symbols: List[str]):
        self.connector = connector
        self.symbols = symbols
        self.logger = logging.getLogger("TickCollector")
        self.is_running = False
        self.callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Sets function to be called for each new tick."""
        self.callback = callback

    def run(self):
        """Continuous polling loop for ticks with stability and rate limiting."""
        self.is_running = True
        self.logger.info(f"Starting tick collection for: {', '.join(self.symbols)}")
        
        last_ticks = {symbol: None for symbol in self.symbols}
        ticks_collected = 0
        start_time = time.time()

        while self.is_running:
            try:
                for symbol in self.symbols:
                    tick = self.connector.get_tick(symbol)
                    if tick:
                        # Deduplication logic: check symbol + millisecond timestamp
                        if last_ticks[symbol] is None or tick['time_msc'] > last_ticks[symbol]:
                            last_ticks[symbol] = tick['time_msc']
                            ticks_collected += 1
                            
                            if self.callback:
                                self.callback(tick)
                        else:
                            # Optional: Log duplicates if needed for debugging at DEBUG level
                            pass
                
                # Report collection rate every 60 seconds
                if time.time() - start_time > 60:
                    rate = ticks_collected / 60
                    self.logger.info(f"Tick collection rate: {rate:.2f} ticks/sec")
                    ticks_collected = 0
                    start_time = time.time()

                # Rate limiter: Balanced polling frequency (approx 66Hz)
                time.sleep(0.015)

            except Exception as e:
                self.logger.error(f"TickCollector loop error: {str(e)}")
                self.logger.info("Attempting automatic collector restart in 5 seconds...")
                time.sleep(5)
                # Ensure connector is still healthy
                if not self.connector.is_connected():
                    self.connector.connect()

    def stop(self):
        self.logger.info("Stopping tick collector...")
        self.is_running = False
