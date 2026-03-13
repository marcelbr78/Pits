import logging
import os
import MetaTrader5 as mt5_lib
from data_engine.mt5_connector import MT5Connector
from data_engine.data_storage import DataStorage

def download_history(symbols: list, count_per_symbol: int = 5000):
    """
    Downloads historical ticks for all symbols and saves them to Parquet.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("HistoricalDownloader")
    
    mt5 = MT5Connector()
    storage = DataStorage()
    
    if not mt5.connect():
        logger.error("Could not connect to MT5 for historical download.")
        return

    logger.info(f"Starting historical download for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            # Ensure symbol is selected
            mt5_lib.symbol_select(symbol, True)
            
            logger.info(f"Fetching {count_per_symbol} ticks for {symbol}...")
            ticks = mt5.get_historical_ticks(symbol, count_per_symbol)
            
            if ticks and len(ticks) > 10:
                logger.info(f"Saving {len(ticks)} ticks for {symbol}...")
                for tick in ticks:
                    storage.save_tick(tick)
                # Force flush at the end of each symbol
                storage.flush(symbol)
                logger.info(f"Successfully bootstrapped {symbol} with historical data.")
            else:
                logger.warning(f"No data retrieved for {symbol}.")
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")

    mt5.shutdown()
    logger.info("Historical download complete.")

if __name__ == "__main__":
    # The 'm' suffix symbols from user's account
    ACTIVE_SYMBOLS = ["USOILm", "BTCUSDm", "ETHUSDm", "XAUUSDm", "UKOILm"]
    download_history(ACTIVE_SYMBOLS, count_per_symbol=200000)
