import logging
import threading
from typing import Dict, Any, List
from data_engine.mt5_connector import MT5Connector
from data_engine.tick_collector import TickCollector
from data_engine.data_storage import DataStorage
from feature_engine.feature_pipeline import FeaturePipeline
from ml_engine.ml_pipeline import MLPipeline
from risk_engine.manager import RiskManager
from execution_engine.trade_executor import TradeExecutor
from execution_engine.position_manager import PositionManager
from execution_engine.execution_pipeline import ExecutionPipeline
from paper_trading.paper_trading_engine import PaperTradingEngine
from paper_trading.trade_logger import TradeLogger
from paper_trading.performance_tracker import PerformanceTracker
from learning_engine.learning_pipeline import LearningPipeline

class PITSOrchestrator:
    """
    Central Brain Orchestrator for PITS.
    Coordinates initialization and communication between all engines.
    """
    def __init__(self, dry_run: bool = False):
        self.logger = self._setup_logger()
        self.engines = {}
        self.is_running = False
        self.dry_run = dry_run
        self.symbols = ["WTI", "XAUUSD", "US500", "DXY", "VIX", "BRENT"]

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("BrainOrchestrator")

    def _on_tick_received(self, tick: Dict[str, Any]):
        """Callback triggered for every new tick."""
        # 1. Store raw tick
        self.storage.save_tick(tick)
        
        # Update simulation if in dry run
        self.execution_pipeline.update_tick(tick)
        
        # 2. Run Learning Engine audit (periodic)
        if hasattr(self, 'learning_pipeline'):
            self.learning_pipeline.run_cycle()
        
        # 3. Process features
        features = self.feature_pipeline.process_tick(tick)
        
        # 3. Intelligence Layer (Phase 1)
        prob_up = self.ml_pipeline.process_features(features)
        recommended_risk = self.risk_manager.calculate_kelly_size(prob_up)
        
        # 4. Execution Layer
        # Note: Position sizing for now uses a placeholder fixed conversion from risk %
        # In Phase 2, this will use actual account equity
        lot_size = recommended_risk * 10 # Example: 0.01 risk -> 0.1 lots
        
        self.execution_pipeline.process_signal(features, prob_up, lot_size)
        
        # 5. Logging (Simplified for live/dry-run)
        if self.dry_run and prob_up != 0.5:
            pass # Execution pipeline handles its own logging

    def initialize_engines(self):
        """Initializes all modular engines."""
        self.logger.info(f"Initializing PITS modular engines (Dry Run: {self.dry_run})...")
        
        # Initialize Components
        self.mt5 = MT5Connector()
        if self.mt5.connect():
            self.storage = DataStorage()
            self.feature_pipeline = FeaturePipeline(self.symbols)
            self.ml_pipeline = MLPipeline(self.symbols)
            self.risk_manager = RiskManager()
            
            # Initialize Execution Engine
            self.executor = TradeExecutor()
            self.position_manager = PositionManager()
            
            # Phase 1: Paper Trading initialization
            self.trade_logger = TradeLogger()
            self.perf_tracker = PerformanceTracker()
            self.paper_engine = PaperTradingEngine(self.trade_logger)
            
            self.execution_pipeline = ExecutionPipeline(
                self.executor, 
                self.position_manager, 
                paper_engine=self.paper_engine,
                live_trading=not self.dry_run
            )
            
            # Initialize Learning Engine
            self.learning_pipeline = LearningPipeline()
            
            self.collector = TickCollector(self.mt5, self.symbols)
            self.collector.set_callback(self._on_tick_received)
            
            self.engines['mt5'] = self.mt5
            self.engines['storage'] = self.storage
            self.engines['collector'] = self.collector
            self.engines['features'] = self.feature_pipeline
            self.engines['ml'] = self.ml_pipeline
            self.engines['risk'] = self.risk_manager
            self.engines['execution'] = self.execution_pipeline
            
            self.logger.info("Full Intelligence & Execution Pipeline (Phase 1) initialized.")
        else:
            self.logger.error("Failed to initialize engines due to MT5 connection error.")

        self.logger.info("Initialization sequence complete.")

    def run(self):
        """Starts the main trading loop and background collectors."""
        self.is_running = True
        self.logger.info("PITS Orchestrator started.")
        
        # Start Tick Collector in a background thread
        if 'collector' in self.engines:
            collector_thread = threading.Thread(target=self.collector.run, daemon=True)
            collector_thread.start()
            self.logger.info("Tick collection background thread started.")

        try:
            while self.is_running:
                # Main orchestration logic (e.g., checking strategy signals)
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops the orchestrator and all engines."""
        self.logger.info("Stopping PITS Orchestrator...")
        if 'collector' in self.engines:
            self.collector.stop()
        
        if 'storage' in self.engines:
            self.logger.info("Flushing final data buffers...")
            self.storage.flush_all()
            
        if 'mt5' in self.engines:
            self.mt5.shutdown()
        self.is_running = False

if __name__ == "__main__":
    orchestrator = PITSOrchestrator()
    orchestrator.initialize_engines()
    orchestrator.run()
