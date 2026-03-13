"""
Dataset Builder V2 — Fase 4.

Versão expandida do DatasetBuilder para treinar modelos
com o vetor de 100+ features do AdvancedFeatureEngine.

Compatível com XGBoost, Random Forest e LSTM.
"""

import pandas as pd
import numpy as np
import glob
import os
import logging
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler


class DatasetBuilderV2:
    def __init__(self, data_path: str = "data/ticks/"):
        self.data_path = data_path
        self.logger = logging.getLogger("DatasetBuilderV2")
        self.scaler = StandardScaler()

    def build_dataset(
        self,
        symbol: str,
        horizon: int = 20,
        min_samples: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Constrói dataset com features avançadas para um símbolo.
        
        Returns: (X_scaled, y, feature_names)
        """
        from feature_engine.advanced_features import AdvancedFeatureEngine
        from feature_engine.ofi_calculator import OFICalculator
        from feature_engine.vwap_calculator import VWAPCalculator
        from feature_engine.volatility_calculator import VolatilityCalculator
        from feature_engine.entropy_calculator import EntropyCalculator

        files = glob.glob(os.path.join(self.data_path, f"{symbol}.parquet"))
        if not files:
            self.logger.warning(f"Sem dados para {symbol}")
            return np.array([]), np.array([]), []

        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df.sort_values('timestamp').reset_index(drop=True)

        if len(df) < min_samples + horizon:
            self.logger.warning(f"Dados insuficientes: {len(df)} < {min_samples + horizon}")
            return np.array([]), np.array([]), []

        self.logger.info(f"Construindo dataset {symbol}: {len(df)} ticks")

        # Inicializa engines
        adv_engine = AdvancedFeatureEngine(symbol)
        ofi_calc   = OFICalculator()
        vwap_calc  = VWAPCalculator()
        vol_calc   = VolatilityCalculator(window_size=100)
        ent_calc   = EntropyCalculator(window_size=50)

        rows = []
        for _, row in df.iterrows():
            tick = {
                'symbol': symbol,
                'bid':    row.get('bid', 0),
                'ask':    row.get('ask', 0),
                'volume': row.get('volume', 1.0),
                'timestamp': row.get('timestamp', 0),
                'flags': row.get('flags', 0),
            }
            tick['spread'] = tick['ask'] - tick['bid']

            base_features = {
                **tick,
                'ofi':        ofi_calc.update(tick),
                'vwap':       vwap_calc.update(tick),
                'volatility': vol_calc.update(tick),
                'entropy':    ent_calc.update(tick),
                'obi':        0.0,   # DOM não disponível em histórico
                'tft':        0.5,
                'vwap_dev_pct': 0.0,
                'vwap_zscore':  0.0,
                'macro_regime': 'NEUTRAL',
                'vol_regime':   'NORMAL_VOL',
                'pre_event_flag': 0,
            }

            adv_feats = adv_engine.compute(base_features)
            rows.append(adv_feats)

        feat_df = pd.DataFrame(rows)
        feat_names = sorted([c for c in feat_df.columns if c not in ('symbol', 'timestamp', 'bid', 'ask')])

        # Labels look-ahead
        mid = (df['bid'] + df['ask']) / 2
        labels = (mid.shift(-horizon) > mid).astype(int)

        feat_df['target'] = labels.values
        feat_df = feat_df.dropna(subset=feat_names + ['target']).iloc[:-horizon]

        X = feat_df[feat_names].fillna(0).values
        y = feat_df['target'].values

        self.logger.info(
            f"Dataset {symbol}: {len(X)} amostras | "
            f"{len(feat_names)} features | "
            f"UP: {y.mean()*100:.1f}%"
        )

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feat_names

    def build_all(
        self,
        symbols: List[str],
        horizon: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Agrega dados de todos os símbolos."""
        Xs, ys, names = [], [], []
        for sym in symbols:
            X, y, feat_names = self.build_dataset(sym, horizon)
            if X.size > 0:
                Xs.append(X)
                ys.append(y)
                if not names:
                    names = feat_names

        if not Xs:
            return np.array([]), np.array([]), []

        return np.vstack(Xs), np.concatenate(ys), names

    def save_scaler(self, path: str = "models/scaler_v2.pkl"):
        try:
            import joblib
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.scaler, path)
        except Exception as e:
            self.logger.error(f"Erro ao salvar scaler: {e}")

    def load_scaler(self, path: str = "models/scaler_v2.pkl"):
        try:
            import joblib
            if os.path.exists(path):
                self.scaler = joblib.load(path)
                return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar scaler: {e}")
        return False
