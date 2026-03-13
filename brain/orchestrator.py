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
from market_intelligence.intelligence_pipeline import IntelligencePipeline
from api.state import SystemState
from api.server import run_server
import os
import time

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
        self.symbols = ["USOILm", "BTCUSDm", "ETHUSDm", "XAUUSDm", "UKOILm"]
        self.state = SystemState()
        self.state.set_live(not dry_run)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("BrainOrchestrator")

    def _on_tick_received(self, tick: Dict[str, Any]):
        """Callback triggered for every new tick."""
        if not self.is_running: return # Stop if paused via API
        
        # 1. Store raw tick
        self.storage.save_tick(tick)
        
        # Update simulation if in dry run
        self.execution_pipeline.update_tick(tick)
        
        # 2. Run Learning Engine audit (periodic)
        if hasattr(self, 'learning_pipeline'):
            self.learning_pipeline.run_cycle()
        
        # 3. Process features
        features = self.feature_pipeline.process_tick(tick)
        
        # 4. Market Intelligence Layer
        market_state = self.market_intelligence.get_market_state(features)
        features['market_state'] = market_state
        
        # 5. Intelligence Layer (Phase 1)
        prob_up = self.ml_pipeline.process_features(features)
        recommended_risk = self.risk_manager.calculate_kelly_size(prob_up)
        
        # 6. Execution Layer
        lot_size = recommended_risk * 10
        self.execution_pipeline.process_signal(features, prob_up, lot_size)
        
        # 7. Update System State for API/Dashboard
        self._update_state(tick, features, prob_up, market_state)

    def _update_state(self, tick, features, prob_up, market_state):
        symbol = tick['symbol'] if isinstance(tick, dict) else (tick.symbol if hasattr(tick, 'symbol') else str(tick))
        
        self.state.update_signal(symbol, prob_up)
        
        if features:
            self.state.update_features(symbol, {
                "ofi": round(float(features.get("ofi", 0)), 4),
                "vwap": round(float(features.get("vwap", 0)), 4),
                "spread": round(float(features.get("spread", 0)), 4),
                "volatility": round(float(features.get("volatility", 0)), 4),
                "entropy": round(float(features.get("entropy", 0)), 4),
            })
        
        if market_state:
            # Handle string response from IntelligencePipeline
            if isinstance(market_state, str):
                parts = market_state.split('_')
                macro = parts[0] if len(parts) > 0 else "UNKNOWN"
                vol = parts[1] if len(parts) > 1 else "UNKNOWN"
                micro = parts[2] if len(parts) > 2 else "UNKNOWN"
            else:
                macro = getattr(market_state, "macro", "UNKNOWN")
                vol = getattr(market_state, "volatility", "UNKNOWN")
                micro = getattr(market_state, "micro", "UNKNOWN")
            
            self.state.update_regime(macro, vol, micro)
        
        self.state.add_log(f"tick {symbol} prob={round(prob_up*100,1)}%")

        # Periodic updates for heavy metadata (positions, metrics)
        now = time.time()
        if not hasattr(self, '_last_slow_update'): self._last_slow_update = 0
        
        if now - self._last_slow_update > 5:
            # Update Positions
            if hasattr(self, 'position_manager'):
                raw_pos = self.position_manager.get_open_positions()
                pos_list = []
                for p in raw_pos:
                    pos_list.append({
                        "symbol": p['symbol'],
                        "type": "BUY" if p['type'] == 0 else "SELL",
                        "entry": p['price_open'],
                        "current": p['price_current'],
                        "pnl": p['profit']
                    })
                self.state.update_positions(pos_list)
            
            # Update Metrics (from Paper Trading)
            if os.path.exists("data/paper_trades.parquet") and hasattr(self, 'perf_tracker'):
                try:
                    df = self.perf_tracker.load_trades("data/paper_trades.parquet")
                    metrics = self.perf_tracker.calculate_metrics(df)
                    self.state.update_metrics(
                        metrics.get('win_rate', 0),
                        metrics.get('sharpe', 0),
                        metrics.get('drawdown', 0),
                        metrics.get('profit_factor', 0)
                    )
                    self.state.update_trades(df.to_dict(orient="records"))
                except:
                    pass
            
            if hasattr(self, 'mt5'):
                self.state.set_mt5_connected(self.mt5.is_connected())
            
            self._last_slow_update = now

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
            
            # Initialize Market Intelligence
            self.market_intelligence = IntelligencePipeline()
            
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
        
        # Start API Server
        api_thread = threading.Thread(
            target=run_server, 
            args=(self.state,), 
            kwargs={"port": 8001}, 
            daemon=True
        )
        api_thread.start()
        self.logger.info("API Server started on port 8001.")

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
