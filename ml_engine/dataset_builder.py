import pandas as pd
import numpy as np
import glob
import os
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any

class DatasetBuilder:
    """
    Transforms historical Parquet tick data into normalized training datasets.
    """
    def __init__(self, data_path: str = "data/ticks/"):
        self.data_path = data_path
        self.logger = logging.getLogger("DatasetBuilder")
        self.scaler = StandardScaler()

    def build_dataset(self, symbol: str, horizon: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a training dataset for a specific symbol.
        1. Loads Parquet files.
        2. Generates features (OFI, Spread, VWAP Dev, Volatility, Entropy).
        3. Generates directional labels using look-ahead N ticks.
        4. Normalizes features.
        """
        file_pattern = os.path.join(self.data_path, f"{symbol}.parquet")
        files = glob.glob(file_pattern)
        
        if not files:
            self.logger.warning(f"No data found for symbol {symbol}")
            return np.array([]), np.array([])

        self.logger.info(f"Loading {len(files)} data files for {symbol}...")
        df_list = [pd.read_parquet(f) for f in files]
        df = pd.concat(df_list).sort_values('timestamp').reset_index(drop=True)

        if len(df) <= horizon:
            self.logger.warning(f"Insufficient data for {symbol} (count: {len(df)})")
            return np.array([]), np.array([])

        # --- Feature Engineering (Simulating FeaturePipeline logic) ---
        # Note: In a real system, we might want to store feature vectors directly,
        # but here we reconstruct them from tick data if needed.
        # Assuming DataStorage saves ticks with some calculated fields or we recalculate.
        
        # OFI can be complex to reconstruct perfectly without state, 
        # but we can approximate or use stored OFI if available.
        # For Phase 1, we assume we need to calculate some fields on the fly.
        
        # Calculate Mid Price
        df['mid'] = (df['bid'] + df['ask']) / 2
        
        # label logic: If mid_price after N ticks is higher than current mid_price
        df['target'] = (df['mid'].shift(-horizon) > df['mid']).astype(int)
        
        # Filter features (Using what's available in the saved Parquet)
        # We need: ofi, spread, vwap_dev, volatility, entropy
        # Let's ensure these columns exist or calculate them.
        required_cols = ['ofi', 'spread', 'vwap', 'volatility', 'entropy', 'bid', 'ask']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
             self.logger.error(f"Missing required columns in Parquet: {missing}")
             return np.array([]), np.array([])

        # Calculate VWAP Deviation
        df['vwap_dev'] = df['mid'] - df['vwap']
        
        feature_cols = ['ofi', 'spread', 'vwap_dev', 'volatility', 'entropy']
        
        # Clean data (remove NaNs from shift and rolling)
        df_clean = df.dropna(subset=feature_cols + ['target']).iloc[: -horizon]
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        self.logger.info(f"Dataset built for {symbol}: {len(X)} samples.")
        self.logger.info(f"Class Balance: {np.mean(y)*100:.2f}% UP")

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def get_all_data(self, symbols: List[str], horizon: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregates data across all symbols."""
        X_all = []
        y_all = []
        
        for symbol in symbols:
            X, y = self.build_dataset(symbol, horizon)
            if X.size > 0:
                X_all.append(X)
                y_all.append(y)
        
        if not X_all:
            return np.array([]), np.array([])
            
        return np.vstack(X_all), np.concatenate(y_all)
