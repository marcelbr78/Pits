import logging
import time
from .performance_analyzer import PerformanceAnalyzer
from .retraining_manager import RetrainingManager

class LearningPipeline:
    """
    Coordinates performance auditing and automated model retraining.
    """
    def __init__(self):
        self.logger = logging.getLogger("LearningPipeline")
        self.analyzer = PerformanceAnalyzer()
        self.manager = RetrainingManager()
        self.last_run_time = time.time()
        self.audit_interval_sec = 3600 # 1 hour default

    def run_cycle(self):
        """Runs one learning cycle: Audit -> Decision -> Action."""
        current_time = time.time()
        if (current_time - self.last_run_time) < self.audit_interval_sec:
            return

        self.logger.info("Starting Learning Pipeline audit cycle...")
        
        # 1. Audit Performance
        metrics = self.analyzer.get_summary()
        
        # 2. Check Decision Logic
        if self.manager.should_retrain(metrics):
            # 3. Action: Retrain
            success = self.manager.trigger_retraining()
            if success:
                self.logger.info("Model updated and saved successfully via Learning Engine.")
            else:
                self.logger.error("Learning cycle completed with retraining failure.")
        else:
            self.logger.info("Learning cycle completed. No retraining needed.")

        self.last_run_time = current_time
