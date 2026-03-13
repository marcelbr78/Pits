import logging

class EnsembleModel:
    """
    Combines predictions from multiple ML models using a weighted average.
    """
    def __init__(self, xgb_weight: float = 0.7, bayes_weight: float = 0.3):
        self.logger = logging.getLogger("EnsembleModel")
        self.xgb_weight = xgb_weight
        self.bayes_weight = bayes_weight

    def predict(self, xgb_prob: float, bayes_prob: float) -> float:
        """
        Calculates the ensemble directional probability.
        Formula: final_prob = (0.7 * xgb_prob) + (0.3 * bayes_prob)
        """
        final_prob = (self.xgb_weight * xgb_prob) + (self.bayes_weight * bayes_prob)
        return round(float(final_prob), 4)
