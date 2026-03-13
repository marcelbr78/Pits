import xgboost as xgb
import numpy as np
import os
import logging
from typing import List, Dict

class XGBoostModel:
    """
    XGBoost model for predicting directional movement probability.
    """
    def __init__(self, model_path: str = "models/xgboost_model.json"):
        self.model_path = model_path
        self.logger = logging.getLogger("XGBoostModel")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Trains the XGBoost model on historical data."""
        if X.size == 0 or y.size == 0:
            self.logger.warning("Empty dataset for XGBoost training.")
            return

        try:
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate metrics
            accuracy = self.model.score(X, y)
            self.logger.info(f"XGBoost model trained. Samples: {len(X)} | Accuracy: {accuracy:.4f}")
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")

    def predict(self, features: List[float]) -> float:
        """Returns the probability of an UP move (class 1)."""
        if not self.is_trained:
            return 0.5

        try:
            X = np.array(features).reshape(1, -1)
            probs = self.model.predict_proba(X)
            return float(probs[0][1]) # Probability of class 1 (UP)
        except Exception as e:
            self.logger.error(f"XGBoost prediction error: {str(e)}")
            return 0.5

    def save_model(self, filepath: str = None):
        """Saves the XGBoost model to a JSON file."""
        path = filepath or self.model_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path)
            self.logger.info(f"XGBoost model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save XGBoost model: {str(e)}")

    def load_model(self, filepath: str = None):
        """Loads the XGBoost model from a JSON file."""
        path = filepath or self.model_path
        if os.path.exists(path):
            try:
                self.model.load_model(path)
                self.is_trained = True
                self.logger.info(f"XGBoost model loaded from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load XGBoost model: {str(e)}")
        else:
            self.logger.warning(f"No XGBoost model found at {path}")
