import numpy as np
from sklearn.naive_bayes import GaussianNB
import logging
from typing import Dict, Any, List, Optional

class BayesianModel:
    """
    Phase 1: Gaussian Naive Bayes prediction model.
    Predicts the probability of the next price move (Up/Down).
    """
    def __init__(self):
        self.logger = logging.getLogger("BayesianModel")
        self.model = GaussianNB()
        self.is_trained = False
        # Features used: ofi, spread, vwap_dev, volatility, entropy
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on historical feature vectors.
        X: Feature matrix
        y: Target direction (1 for Up, 0 for Down)
        """
        if X.size == 0 or y.size == 0:
            self.logger.warning("Empty dataset provided for training.")
            return

        try:
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate and log accuracy
            accuracy = self.model.score(X, y)
            balance = np.mean(y)
            self.logger.info(f"Bayesian model trained. Samples: {len(X)} | Accuracy: {accuracy:.4f} | Balance: {balance:.2f}")
        except Exception as e:
            self.logger.error(f"Error training Bayesian model: {str(e)}")

    def save_model(self, filepath: str = "models/bayesian_model.pkl"):
        """Saves the trained model to disk."""
        import joblib
        import os
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str = "models/bayesian_model.pkl"):
        """Loads a model from disk."""
        import joblib
        import os
        if os.path.exists(filepath):
            try:
                self.model = joblib.load(filepath)
                self.is_trained = True
                self.logger.info(f"Model loaded from {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
        else:
            self.logger.warning(f"No model file found at {filepath}")

    def predict(self, features: List[float]) -> Dict[str, float]:
        """
        Returns directional probabilities.
        """
        if not self.is_trained:
            # Fallback for untrained model: Neutral 50/50
            return {"probability_up": 0.5, "probability_down": 0.5}

        try:
            X = np.array(features).reshape(1, -1)
            probs = self.model.predict_proba(X)[0]
            
            return {
                "probability_up": probs[1],
                "probability_down": probs[0]
            }
        except Exception as e:
            self.logger.error(f"Error during Bayesian prediction: {str(e)}")
            return {"probability_up": 0.5, "probability_down": 0.5}
