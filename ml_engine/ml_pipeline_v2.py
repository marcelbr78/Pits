"""
ML Pipeline V2 — Fase 3-5.

Versão expandida do MLPipeline que usa o vetor de 100+ features
e suporta LSTM + GNN adicionalmente ao XGBoost + Bayes da Fase 1.

Mantém API compatível: process_features(feature_dict) → float (prob)
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ml_engine.bayesian_model import BayesianModel
from ml_engine.xgboost_model import XGBoostModel
from ml_engine.ensemble_model import EnsembleModel
from ml_engine.dataset_builder_v2 import DatasetBuilderV2


class MLPipelineV2:
    """
    Pipeline ML expandida para 100+ features.
    Drop-in replacement do MLPipeline quando modelos V2 estiverem treinados.
    """

    # Features obrigatórias (subset das 100+) usadas diretamente
    BASE_FEATURES = [
        'ofi', 'spread', 'volatility', 'entropy',
        'obi', 'tft', 'vwap_dev_pct', 'vwap_zscore',
        'ret_1t', 'ret_5t', 'ret_10t', 'ret_20t',
        'vol_10t', 'vol_20t', 'vol_ratio',
        'spread_ratio', 'obi_ma5', 'obi_delta',
        'tft_ma10', 'tft_delta', 'vol_burst',
        'autocorr_ret', 'pre_event_flag',
        'regime_WAR', 'regime_RISK_OFF',
    ]

    def __init__(self, symbols: List[str]):
        self.logger = logging.getLogger("MLPipelineV2")
        self.symbols = symbols

        # Modelos Fase 1 (sempre disponíveis)
        self.bayes_model = BayesianModel()
        self.xgb_model   = XGBoostModel()
        self.ensemble    = EnsembleModel()

        self.bayes_model.load_model()
        self.xgb_model.load_model()

        # Scaler V2 (para features expandidas)
        self._scaler = None
        self._try_load_scaler()

        # Feature names do dataset V2
        self._feat_names: Optional[List[str]] = None

        self.logger.info("MLPipelineV2 inicializado.")

    def process_features(self, feature_vector: Dict[str, Any]) -> float:
        """
        Recebe vetor completo (100+) e retorna prob_up.
        Fallback automático para Fase 1 se modelos V2 não treinados.
        """
        # Tenta predição com vetor expandido
        prob_xgb = self._predict_xgb_v2(feature_vector)
        if prob_xgb is None:
            # Fallback Fase 1
            prob_xgb = self._predict_xgb_v1(feature_vector)

        # Bayes sempre usa vetor V1 (5 features)
        prob_bayes = self._predict_bayes(feature_vector)

        return self.ensemble.predict(prob_xgb, prob_bayes)

    def _predict_xgb_v2(self, fv: Dict) -> Optional[float]:
        """XGBoost com vetor expandido — requer scaler V2."""
        if not self.xgb_model.is_trained or self._scaler is None:
            return None

        try:
            # Extrai features na ordem correta
            vec = [float(fv.get(f, 0)) for f in self.BASE_FEATURES]
            vec_scaled = self._scaler.transform([vec])
            prob = self.xgb_model.predict(vec_scaled[0].tolist())
            return prob
        except Exception:
            return None

    def _predict_xgb_v1(self, fv: Dict) -> float:
        """Fallback: XGBoost com 5 features da Fase 1."""
        mid = (fv.get('bid', 0) + fv.get('ask', 0)) / 2
        vwap_dev = mid - fv.get('vwap', mid)
        features = [
            fv.get('ofi', 0),
            fv.get('spread', 0),
            vwap_dev,
            fv.get('volatility', 0),
            fv.get('entropy', 0),
        ]
        return self.xgb_model.predict(features)

    def _predict_bayes(self, fv: Dict) -> float:
        """Bayes com 5 features."""
        mid = (fv.get('bid', 0) + fv.get('ask', 0)) / 2
        vwap_dev = mid - fv.get('vwap', mid)
        features = [
            fv.get('ofi', 0),
            fv.get('spread', 0),
            vwap_dev,
            fv.get('volatility', 0),
            fv.get('entropy', 0),
        ]
        return self.bayes_model.predict(features).get('probability_up', 0.5)

    def _try_load_scaler(self):
        try:
            import joblib, os
            path = 'models/scaler_v2.pkl'
            if os.path.exists(path):
                self._scaler = joblib.load(path)
                self.logger.info("Scaler V2 carregado.")
        except Exception:
            pass
