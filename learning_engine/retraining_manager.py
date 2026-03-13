import logging
import os
import subprocess
import sys

class RetrainingManager:
    """
    Determines if ML model needs retraining based on performance metrics.
    """
    def __init__(self, win_rate_threshold: float = 0.45, sharpe_threshold: float = 0.5):
        self.logger = logging.getLogger("RetrainingManager")
        self.win_rate_threshold = win_rate_threshold
        self.sharpe_threshold = sharpe_threshold
        self.training_script = "train_pits_model.py"

    def should_retrain(self, metrics: dict) -> bool:
        """Evaluates metrics against thresholds."""
        if not metrics:
            return False
            
        win_rate = metrics.get("win_rate", 1.0)
        sharpe = metrics.get("sharpe_ratio", 1.0)
        total_trades = metrics.get("total_trades", 0)

        # Only retrain if we have a minimum amount of data to be statistically significant
        if total_trades < 50:
            return False

        if win_rate < self.win_rate_threshold or sharpe < self.sharpe_threshold:
            self.logger.warning(f"Retraining Triggered: WinRate={win_rate:.2f}, Sharpe={sharpe:.2f}")
            return True
            
        return False

    def trigger_retraining(self):
        """Executes the training script."""
        self.logger.info("Executing model retraining pipeline...")
        try:
            # Run the training script as a subprocess
            result = subprocess.run(
                [sys.executable, self.training_script], 
                capture_output=True, 
                text=True,
                check=True
            )
            self.logger.info(f"Retraining complete. Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Retraining failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during retraining: {str(e)}")
            return False
