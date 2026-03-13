from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class MLModel(ABC):
    """Abstract base class for all predictive models in PITS."""
    
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Returns (predicted_value, confidence_score)."""
        pass

class BayesianInference(MLModel):
    """Phase 1: Simple Bayesian Inference model."""
    def train(self, X, y): pass
    def predict(self, features): return 0.5, 0.6

class XGBoostModel(MLModel):
    """Phase 4: Advanced XGBoost prediction."""
    def train(self, X, y): pass
    def predict(self, features): return 0.5, 0.8
