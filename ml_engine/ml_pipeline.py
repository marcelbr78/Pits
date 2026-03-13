import logging
from typing import Dict, Any, List
from .bayesian_model import BayesianModel
from .xgboost_model import XGBoostModel
from .ensemble_model import EnsembleModel

class MLPipeline:
    """
    Orchestrates the machine learning workflow using an ensemble of models.
    """
    def __init__(self, symbols: List[str]):
        self.logger = logging.getLogger("MLPipeline")
        self.symbols = symbols
        
        # Initialize sub-models
        self.bayes_model = BayesianModel()
        self.xgb_model = XGBoostModel()
        self.ensemble = EnsembleModel()
        
        # Load pre-trained models
        self.bayes_model.load_model()
        self.xgb_model.load_model()

    def process_features(self, feature_vector: Dict[str, Any]) -> float:
        """
        Receives feature vector, runs ensemble prediction, and returns final probability.
        """
        # Calculate mid price and vwap deviation
        mid = (feature_vector['bid'] + feature_vector['ask']) / 2
        vwap_dev = mid - feature_vector['vwap']
        
        # Features in required order: [ofi, spread, vwap_deviation, volatility, entropy]
        features = [
            feature_vector['ofi'],
            feature_vector['spread'],
            vwap_dev,
            feature_vector['volatility'],
            feature_vector['entropy']
        ]

        # 1. Individual predictions
        prob_bayes = self.bayes_model.predict(features)['probability_up']
        prob_xgb = self.xgb_model.predict(features)
        
        # 2. Ensemble calculation
        final_probability = self.ensemble.predict(prob_xgb, prob_bayes)
        
        return final_probability
