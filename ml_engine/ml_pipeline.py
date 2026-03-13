import logging
from typing import Dict, Any, List
from .bayesian_model import BayesianModel

class MLPipeline:
    """
    Orchestrates the machine learning workflow.
    Receives feature vectors and produces trading signals.
    """
    def __init__(self, symbols: List[str]):
        self.logger = logging.getLogger("MLPipeline")
        self.models: Dict[str, BayesianModel] = {symbol: BayesianModel() for symbol in symbols}

    def process_features(self, feature_vector: Dict[str, Any]) -> float:
        """
        Processes a feature vector and returns the probability of an 'Up' move.
        """
        symbol = feature_vector['symbol']
        if symbol not in self.models:
            return 0.5

        # Extract features in the required order:
        # [ofi, spread, vwap_deviation, volatility, entropy]
        # Note: vwap_deviation is calculated here as price - vwap
        vwap_dev = (feature_vector['bid'] + feature_vector['ask']) / 2 - feature_vector['vwap']
        
        features = [
            feature_vector['ofi'],
            feature_vector['spread'],
            vwap_dev,
            feature_vector['volatility'],
            feature_vector['entropy']
        ]

        prediction = self.models[symbol].predict(features)
        return prediction['probability_up']
