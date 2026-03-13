import pandas as pd
import os
import logging
from typing import Dict, Any
import time

class TradeLogger:
    """
    Handles persisting simulated paper trades to Parquet.
    """
    def __init__(self, storage_path: str = "data/paper_trades.parquet"):
        self.storage_path = storage_path
        self.logger = logging.getLogger("TradeLogger")
        self.trades_list = []
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Appends a completed trade to the buffer and saves to disk."""
        self.trades_list.append(trade_data)
        self._save_to_disk()
        self.logger.info(f"Simulated trade logged: {trade_data['symbol']} | PnL: {trade_data['profit_loss']:.4f}")

    def _save_to_disk(self):
        """Saves all logged trades to a Parquet file."""
        try:
            df = pd.DataFrame(self.trades_list)
            # Ensure proper types
            df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'], unit='s', errors='ignore')
            df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'], unit='s', errors='ignore')
            
            df.to_parquet(self.storage_path, engine='pyarrow', index=False)
        except Exception as e:
            self.logger.error(f"Failed to save paper trades to disk: {str(e)}")

    def get_all_trades(self) -> pd.DataFrame:
        """Returns a DataFrame of all logged trades."""
        if not self.trades_list:
            if os.path.exists(self.storage_path):
                return pd.read_parquet(self.storage_path)
            return pd.DataFrame()
        return pd.DataFrame(self.trades_list)
