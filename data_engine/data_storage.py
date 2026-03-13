import pandas as pd
import os
import logging
from typing import Dict, Any

class DataStorage:
    """
    Handles persisting tick data to Parquet format.
    """
    def __init__(self, storage_path: str = "data/ticks", buffer_size: int = 500):
        self.storage_path = storage_path
        self.buffer_size = buffer_size
        self.logger = logging.getLogger("DataStorage")
        self.buffers: Dict[str, list] = {}
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _validate_tick(self, tick_data: Dict[str, Any]) -> bool:
        """Validates tick data integrity."""
        try:
            if tick_data.get('ask', 0) < tick_data.get('bid', 0):
                self.logger.warning(f"Rejected tick {tick_data['symbol']}: Ask ({tick_data['ask']}) < Bid ({tick_data['bid']})")
                return False
            if 'timestamp' not in tick_data or 'time_msc' not in tick_data:
                self.logger.warning(f"Rejected tick {tick_data['symbol']}: Missing timestamp")
                return False
            
            # Ensure volume is at least 0
            if tick_data.get('volume', -1) < 0:
                tick_data['volume'] = 0.0
                
            return True
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False

    def save_tick(self, tick_data: Dict[str, Any]):
        """Buffers a tick and flushes to parquet if buffer limit reached."""
        if not self._validate_tick(tick_data):
            return

        symbol = tick_data['symbol']
        
        # Calculate Spread
        tick_data['spread'] = round(tick_data['ask'] - tick_data['bid'], 6)
        
        # Standardize timestamp to datetime
        tick_data['timestamp_dt'] = pd.to_datetime(tick_data['time_msc'], unit='ms')

        if symbol not in self.buffers:
            self.buffers[symbol] = []
        
        self.buffers[symbol].append(tick_data)
        
        if len(self.buffers[symbol]) >= self.buffer_size:
            self.flush(symbol)

    def flush_all(self):
        """Flushes all symbol buffers to disk."""
        for symbol in list(self.buffers.keys()):
            self.flush(symbol)

    def flush(self, symbol: str):
        """Flushes a specific symbol's buffer to parquet."""
        if not self.buffers.get(symbol):
            return

        file_path = os.path.join(self.storage_path, f"{symbol}.parquet")
        
        # Convert buffer to DataFrame and standardize schema
        df_new = pd.DataFrame(self.buffers[symbol])
        
        # Define exact schema for standardization
        schema_cols = ['symbol', 'timestamp', 'bid', 'ask', 'spread', 'volume']
        if 'timestamp' in df_new.columns and 'timestamp_dt' in df_new.columns:
            df_new = df_new.drop(columns=['timestamp'])
            
        df_new = df_new.rename(columns={'timestamp_dt': 'timestamp'})
        df_new = df_new[schema_cols].astype({
            'symbol': 'string',
            'bid': 'float64',
            'ask': 'float64',
            'spread': 'float64',
            'volume': 'float64'
        })

        try:
            if os.path.exists(file_path):
                # Using fastparquet or pyarrow with append logic if possible, 
                # but standard concat+write is safer for small batches in pandas
                df_old = pd.read_parquet(file_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_parquet(file_path, engine='pyarrow', index=False)
            else:
                df_new.to_parquet(file_path, engine='pyarrow', index=False)
            
            self.logger.info(f"Flushed {len(self.buffers[symbol])} ticks for {symbol} to {file_path}")
            self.buffers[symbol] = [] # Clear buffer
        except Exception as e:
            self.logger.error(f"Error flushing ticks for {symbol}: {str(e)}")

    def get_tick_history(self, symbol: str) -> pd.DataFrame:
        """Loads tick history from parquet."""
        file_path = os.path.join(self.storage_path, f"{symbol}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return pd.DataFrame()
