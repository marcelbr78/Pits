"""
Orchestrator Final — Fases 2-5.

Integra todos os componentes de todas as fases:

Fase 1 (existente):
  - Tick collector, Data storage
  - OFI, VWAP, Shannon, Volatility
  - XGBoost + Bayes ensemble
  - Paper trading engine

Fase 2 (adicionada):
  - OBI 10 níveis
  - Trade Flow Toxicity
  - Lag features cross-asset
  - Calendário econômico

Fase 3 (adicionada):
  - Pattern Library (pandemia, guerra, etc)
  - ATR Calculator
  - TP/SL dinâmicos

Fase 4 (adicionada):
  - 100+ features
  - Portfolio Risk Engine
  - Monte Carlo
  - Anomaly Detector

Fase 5 (adicionada):
  - LSTM
  - GNN
  - Ensemble V2 com pesos dinâmicos

Uso:
  python run_pits_final.py           # paper trading
  python run_pits_final.py --live    # live trading (cuidado!)
"""

import logging
import threading
import time
from typing import Dict, Any

from data_engine.mt5_connector import MT5Connector
from data_engine.tick_collector import TickCollector
from data_engine.data_storage import DataStorage

# Feature engines
from feature_engine.feature_pipeline_v2 import FeaturePipelineV2
from feature_engine.advanced_features import AdvancedFeatureEngine
from feature_engine.atr_calculator import ATRCalculator

# Market intelligence
from market_intelligence.intelligence_pipeline_v2 import IntelligencePipelineV2
from market_intelligence.pattern_library import PatternLibrary
from market_intelligence.anomaly_detector import AnomalyDetector

# ML engines
from ml_engine.ml_pipeline import MLPipeline
from ml_engine.lstm_model import LSTMModel
from ml_engine.gnn_model import TradingGNN
from ml_engine.ensemble_v2 import EnsembleV2

# Risk engines
from risk_engine.manager import RiskManager
from risk_engine.portfolio_risk import PortfolioRiskEngine
from risk_engine.monte_carlo import MonteCarloSimulator

# Execution
from execution_engine.trade_executor import TradeExecutor
from execution_engine.position_manager import PositionManager
from execution_engine.execution_pipeline import ExecutionPipeline

# Paper trading
from paper_trading.paper_trading_engine import PaperTradingEngine
from paper_trading.trade_logger import TradeLogger
from paper_trading.performance_tracker import PerformanceTracker

# Learning
from learning_engine.learning_pipeline import LearningPipeline

# API
from api.state import SystemState
from api.server import run_server


class PITSOrchestratorFinal:
    """
    Orchestrator completo — todas as fases.
    """

    def __init__(self, dry_run: bool = True):
        self.logger = self._setup_logger()
        self.dry_run  = dry_run
        self.is_running = False
        self.symbols  = ["USOILm", "BTCUSDm", "ETHUSDm", "XAUUSDm", "UKOILm"]
        self.state    = SystemState()
        self.state.set_live(not dry_run)
        self._tick_count = 0
        self._start_time = time.time()
        self.engines: Dict[str, Any] = {}

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("PITSFinal")

    # ─────────────────────────────────────────────────────────
    def initialize_engines(self):
        self.logger.info(f"Inicializando PITS Final (dry_run={self.dry_run})...")

        # MT5
        self.mt5 = MT5Connector()
        if not self.mt5.connect():
            self.logger.error("MT5 falhou. Abortando.")
            return

        # Data
        self.storage = DataStorage()

        # Feature Engines
        self.feature_pipeline = FeaturePipelineV2(self.symbols)
        self.adv_engines = {sym: AdvancedFeatureEngine(sym) for sym in self.symbols}
        self.atr_calcs   = {sym: ATRCalculator() for sym in self.symbols}

        # Market Intelligence
        self.market_intelligence = IntelligencePipelineV2()
        self.pattern_library     = PatternLibrary()
        self.anomaly_detector    = AnomalyDetector()

        # ML
        self.ml_pipeline = MLPipeline(self.symbols)
        self.lstm_model  = LSTMModel(n_features=25, seq_length=60)
        self.gnn_model   = TradingGNN()
        self.ensemble_v2 = EnsembleV2()

        # Risk
        self.risk_manager    = RiskManager()
        self.portfolio_risk  = PortfolioRiskEngine(capital=30.0)
        self.monte_carlo     = MonteCarloSimulator(n_simulations=1000)

        # Execution
        self.executor         = TradeExecutor()
        self.position_manager = PositionManager()
        self.trade_logger     = TradeLogger()
        self.perf_tracker     = PerformanceTracker()
        self.paper_engine     = PaperTradingEngine(self.trade_logger)
        self.execution_pipeline = ExecutionPipeline(
            self.executor, self.position_manager,
            paper_engine=self.paper_engine,
            live_trading=not self.dry_run,
        )

        # Collector
        self.collector = TickCollector(self.mt5, self.symbols)
        self.collector.set_callback(self._on_tick_received)

        self.logger.info("PITS Final inicializado — todas as fases ativas.")

    # ─────────────────────────────────────────────────────────
    def _on_tick_received(self, tick: Dict[str, Any]):
        if not self.is_running:
            return

        self._tick_count += 1
        elapsed = time.time() - self._start_time
        symbol  = tick['symbol']

        # Armazena
        self.storage.save_tick(tick)
        self.execution_pipeline.update_tick(tick)

        # ATR
        atr = self.atr_calcs[symbol].update(tick)
        if self.atr_calcs[symbol].should_pause(atr):
            self.state.add_log(f"[{symbol}] ATR={atr:.2f} > 1.50 — PAUSA")
            return

        # Features V2
        features = self.feature_pipeline.process_tick(tick)

        # Features avançadas 100+
        adv_feats = self.adv_engines[symbol].compute(features)
        features.update(adv_feats)

        # Anomaly check
        anomaly = self.anomaly_detector.update(features)
        if anomaly['is_anomaly']:
            self.state.add_log(f"[{symbol}] ANOMALIA: {anomaly['reason']}")
            return

        # Market Intelligence V2
        market_state = self.market_intelligence.get_market_state(features)
        features['macro_regime']    = market_state.macro
        features['vol_regime']      = market_state.volatility_regime
        features['micro_regime']    = market_state.trend
        features['pre_event_flag']  = market_state.calendar_modifier.get('pre_event_flag', 0)

        # Pattern Library
        pattern = self.pattern_library.match(
            macro_regime=market_state.macro,
            vol_regime=market_state.volatility_regime,
            obi=features.get('obi', 0),
            vwap_dev_pct=features.get('vwap_dev_pct', 0),
        )
        tp_dist, sl_dist = self.pattern_library.get_dynamic_tp_sl(pattern, atr)

        # ML — ensemble completo
        prob_xgb   = self.ml_pipeline.process_features(features)
        prob_bayes = prob_xgb  # já é ensemble interno F1
        prob_lstm  = self.lstm_model.predict()
        prob_gnn   = self.gnn_model.predict(symbol, market_state.macro)

        # Atualiza GNN node
        node_feats = [
            features.get('obi', 0),
            features.get('tft', 0.5),
            features.get('volatility', 0),
            features.get('entropy', 0),
            features.get('vwap_dev_pct', 0),
            features.get('ret_5t', 0),
            features.get('spread_ratio', 1),
            prob_xgb,
        ]
        self.gnn_model.update_node(symbol, node_feats)

        # Atualiza LSTM buffer
        lstm_feat_vec = [adv_feats.get(k, 0) for k in sorted(adv_feats.keys())[:25]]
        self.lstm_model.update_buffer(lstm_feat_vec)

        # Ensemble V2 com pesos dinâmicos
        ensemble_result = self.ensemble_v2.predict(
            xgb_prob=prob_xgb,
            bayes_prob=prob_bayes,
            lstm_prob=prob_lstm if self.lstm_model.is_trained else None,
            gnn_prob=prob_gnn   if self.gnn_model.is_trained else None,
            macro_regime=market_state.macro,
            micro_regime=market_state.trend,
        )
        prob_final = ensemble_result['probability']

        # Portfolio Risk check
        cal = market_state.calendar_modifier
        if not cal.get('should_pause', False):
            risk_check = self.portfolio_risk.can_trade(
                symbol, 0.01, market_state.macro
            )
            if risk_check['allowed']:
                prob_threshold = cal.get('prob_threshold', 0.75)
                self.execution_pipeline.process_signal(
                    features, prob_final, 0.01,
                    prob_threshold_override=prob_threshold
                )

        # Atualiza dashboard
        self._update_state(symbol, features, prob_final, market_state, elapsed, ensemble_result, atr, pattern)

    # ─────────────────────────────────────────────────────────
    def _update_state(self, symbol, features, prob, market_state, elapsed, ensemble, atr, pattern):
        self.state.update_signal(symbol, prob)

        self.state.update_features(symbol, {
            # Fase 1
            'ofi':        round(float(features.get('ofi', 0)), 4),
            'vwap':       round(float(features.get('vwap', 0)), 4),
            'spread':     round(float(features.get('spread', 0)), 6),
            'volatility': round(float(features.get('volatility', 0)), 6),
            'entropy':    round(float(features.get('entropy', 0)), 4),
            # Fase 2
            'obi':          round(float(features.get('obi', 0)), 4),
            'tft':          round(float(features.get('tft', 0.5)), 4),
            'vwap_dev_pct': round(float(features.get('vwap_dev_pct', 0)), 6),
            'vwap_zscore':  round(float(features.get('vwap_zscore', 0)), 4),
            # Fase 3
            'atr':          round(float(atr), 5),
            'pattern':      pattern['name'] if pattern else 'N/A',
            'pattern_sim':  pattern['similarity'] if pattern else 0.0,
            # Fase 4-5
            'ensemble_conf':    ensemble.get('confidence', 0),
            'models_active':    ensemble.get('models_active', 0),
            'macro_regime':     market_state.macro,
            'pre_event':        features.get('pre_event_flag', 0),
        })

        self.state.update_regime(
            macro=market_state.macro,
            volatility=market_state.volatility_regime,
            micro=market_state.trend,
        )

        self.state.ticks_per_second = round(self._tick_count / max(1, elapsed), 2)

        if self._tick_count % 10 == 0:
            try:
                m = self.perf_tracker.get_summary()
                self.state.update_metrics(
                    m.get('win_rate', 0), m.get('sharpe_ratio', 0),
                    m.get('max_drawdown', 0), m.get('profit_factor', 0)
                )
                self.state.update_positions(self.position_manager.get_open_positions())
            except Exception:
                pass

        try:
            with self.state.lock:
                self.state.next_event = self.market_intelligence.get_next_event()
        except Exception:
            pass

        self.state.set_mt5_connected(self.mt5.is_connected())
        self.state.set_running(True)

        if self._tick_count % 100 == 0:
            self.state.add_log(
                f"[FINAL] TPS:{self.state.ticks_per_second} | "
                f"Regime:{market_state.combined} | "
                f"Prob:{prob:.2f} | "
                f"Models:{ensemble.get('models_active', 0)}/4"
            )

    # ─────────────────────────────────────────────────────────
    def run(self):
        self.is_running = True
        self.logger.info("PITS Final iniciado.")

        # API
        threading.Thread(
            target=run_server, args=(self.state,),
            kwargs={"port": 8001}, daemon=True
        ).start()
        self.logger.info("API Server na porta 8001.")

        # ngrok
        try:
            import requests
            r = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            if r.status_code == 200:
                tunnels = r.json().get('tunnels', [])
                if tunnels:
                    url = tunnels[0]['public_url']
                    self.state.add_log(f"Public URL: {url}")
        except Exception:
            pass

        # Collector
        threading.Thread(target=self.collector.run, daemon=True).start()
        self.logger.info("Tick collector iniciado.")

        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.logger.info("Parando PITS Final...")
        try:
            self.collector.stop()
        except Exception:
            pass
        try:
            self.storage.flush_all()
        except Exception:
            pass
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        self.is_running = False


if __name__ == "__main__":
    import sys
    live = "--live" in sys.argv
    orc = PITSOrchestratorFinal(dry_run=not live)
    orc.initialize_engines()
    orc.run()
