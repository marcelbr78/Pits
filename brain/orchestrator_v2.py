"""
Orchestrator V2 — Fase 2.

Diferenças do V1:
  - Usa FeaturePipelineV2 (OBI + TFT + lag features + VWAP deviation)
  - Usa IntelligencePipelineV2 (3 camadas de regime + calendário)
  - Calendário econômico integrado no threshold de execução
  - SystemState atualizado com dados ricos da Fase 2
  - Prob threshold dinâmico baseado no calendário

Para usar: substitua PITSOrchestrator por PITSOrchestratorV2
em run_system_test.py e train_pits_model.py
"""

import logging
import threading
import time
from typing import Dict, Any

from data_engine.mt5_connector import MT5Connector
from data_engine.tick_collector import TickCollector
from data_engine.data_storage import DataStorage
from feature_engine.feature_pipeline_v2 import FeaturePipelineV2
from ml_engine.ml_pipeline import MLPipeline
from risk_engine.manager import RiskManager
from execution_engine.trade_executor import TradeExecutor
from execution_engine.position_manager import PositionManager
from execution_engine.execution_pipeline import ExecutionPipeline
from paper_trading.paper_trading_engine import PaperTradingEngine
from paper_trading.trade_logger import TradeLogger
from paper_trading.performance_tracker import PerformanceTracker
from learning_engine.learning_pipeline import LearningPipeline
from market_intelligence.intelligence_pipeline_v2 import IntelligencePipelineV2
from api.state import SystemState
from api.server import run_server


class PITSOrchestratorV2:
    """
    Orchestrator Fase 2.
    Drop-in replacement do PITSOrchestrator.
    """

    def __init__(self, dry_run: bool = False):
        self.logger = self._setup_logger()
        self.engines = {}
        self.is_running = False
        self.dry_run = dry_run
        self.symbols = ["USOILm", "BTCUSDm", "ETHUSDm", "XAUUSDm", "UKOILm"]
        self.state = SystemState()
        self.state.set_live(not dry_run)
        self._tick_count = 0
        self._start_time = time.time()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("PITSOrchestratorV2")

    def _on_tick_received(self, tick: Dict[str, Any]):
        """Callback para cada tick — versão Fase 2."""
        if not self.is_running:
            return

        self._tick_count += 1
        elapsed = time.time() - self._start_time
        symbol = tick['symbol']

        # 1. Armazena tick bruto
        self.storage.save_tick(tick)

        # 2. Atualiza simulação paper trading
        self.execution_pipeline.update_tick(tick)

        # 3. Learning engine (periódico)
        if hasattr(self, 'learning_pipeline') and self._tick_count % 500 == 0:
            self.learning_pipeline.run_cycle()

        # 4. Features V2 — vetor expandido com OBI, TFT, lag features
        features = self.feature_pipeline.process_tick(tick)

        # 5. Market Intelligence V2 — 3 camadas + calendário
        market_state = self.market_intelligence.get_market_state(features)
        features['market_state'] = market_state
        features['macro_regime'] = market_state.macro
        features['vol_regime']   = market_state.volatility_regime
        features['micro_regime'] = market_state.trend
        features['pre_event_flag'] = market_state.calendar_modifier.get('pre_event_flag', 0)

        # 6. ML — probabilidade
        prob_up = self.ml_pipeline.process_features(features)

        # 7. Risk — lote
        recommended_risk = self.risk_manager.calculate_kelly_size(prob_up)
        lot_size = recommended_risk * 10

        # 8. Threshold dinâmico baseado no calendário
        cal = market_state.calendar_modifier
        if cal.get('should_pause', False):
            # Muito próximo de evento — não opera
            self.state.add_log(f"PAUSA pré-evento: {cal['reason']}")
        else:
            prob_threshold = cal.get('prob_threshold', 0.75)
            self.execution_pipeline.process_signal(
                features, prob_up, lot_size,
                prob_threshold_override=prob_threshold
            )

        # 9. Atualiza SystemState para o dashboard
        self._update_state(symbol, features, prob_up, market_state, elapsed)

    def _update_state(
        self, symbol, features, prob_up, market_state, elapsed
    ):
        """Alimenta o dashboard com dados em tempo real."""

        self.state.update_signal(symbol, prob_up)

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
            'tft_signal':   features.get('tft_signal', 'NEUTRAL'),
            'vwap_dev_pct': round(float(features.get('vwap_dev_pct', 0)), 6),
            'vwap_zscore':  round(float(features.get('vwap_zscore', 0)), 4),
            'dom_available': features.get('dom_available', False),
            'macro_regime': features.get('macro_regime', 'UNKNOWN'),
            'pre_event':    features.get('pre_event_flag', 0),
        })

        self.state.update_regime(
            macro=market_state.macro,
            volatility=market_state.volatility_regime,
            micro=market_state.trend
        )

        self.state.ticks_per_second = round(
            self._tick_count / max(1, elapsed), 2
        )

        if self._tick_count % 10 == 0:
            try:
                metrics = self.perf_tracker.get_summary()
                self.state.update_metrics(
                    metrics.get('win_rate', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('profit_factor', 0)
                )
                self.state.update_positions(
                    self.position_manager.get_open_positions()
                )
            except Exception:
                pass

        # Calendário no state
        try:
            next_event = self.market_intelligence.get_next_event()
            with self.state.lock:
                self.state.next_event = next_event
        except Exception:
            pass

        self.state.set_mt5_connected(self.mt5.is_connected())
        self.state.set_running(True)

        if self._tick_count % 100 == 0:
            self.state.add_log(
                f"[V2] {self._tick_count} ticks | "
                f"TPS:{self.state.ticks_per_second} | "
                f"Regime:{market_state.combined}"
            )

    def initialize_engines(self):
        """Inicializa todos os engines — Fase 2."""
        self.logger.info(f"Inicializando PITS V2 (dry_run={self.dry_run})...")

        self.mt5 = MT5Connector()
        if not self.mt5.connect():
            self.logger.error("Falha na conexão MT5.")
            return

        self.storage         = DataStorage()
        self.feature_pipeline = FeaturePipelineV2(self.symbols)   # V2
        self.ml_pipeline     = MLPipeline(self.symbols)
        self.risk_manager    = RiskManager()
        self.executor        = TradeExecutor()
        self.position_manager = PositionManager()
        self.trade_logger    = TradeLogger()
        self.perf_tracker    = PerformanceTracker()
        self.paper_engine    = PaperTradingEngine(self.trade_logger)
        self.execution_pipeline = ExecutionPipeline(
            self.executor,
            self.position_manager,
            paper_engine=self.paper_engine,
            live_trading=not self.dry_run
        )
        self.market_intelligence = IntelligencePipelineV2()   # V2
        self.collector = TickCollector(self.mt5, self.symbols)
        self.collector.set_callback(self._on_tick_received)

        self.engines = {
            'mt5': self.mt5, 'storage': self.storage,
            'collector': self.collector, 'features': self.feature_pipeline,
            'ml': self.ml_pipeline, 'risk': self.risk_manager,
            'execution': self.execution_pipeline,
        }

        self.logger.info("PITS V2 inicializado — Fase 2 ativa.")

    def run(self):
        """Inicia o sistema."""
        self.is_running = True
        self.logger.info("PITS V2 iniciado.")

        # API Server
        api_thread = threading.Thread(
            target=run_server,
            args=(self.state,),
            kwargs={"port": 8001},
            daemon=True
        )
        api_thread.start()
        self.logger.info("API Server na porta 8001.")

        # ngrok URL (se disponível)
        try:
            import requests
            r = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            if r.status_code == 200:
                tunnels = r.json().get('tunnels', [])
                if tunnels:
                    url = tunnels[0].get('public_url')
                    self.logger.info(f"PUBLIC URL: {url}")
                    self.state.add_log(f"Public URL: {url}")
        except Exception:
            pass

        # Tick Collector
        if 'collector' in self.engines:
            t = threading.Thread(target=self.collector.run, daemon=True)
            t.start()
            self.logger.info("Tick collector iniciado.")

        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Para o sistema com flush seguro."""
        self.logger.info("Parando PITS V2...")
        if 'collector' in self.engines:
            self.collector.stop()
        if 'storage' in self.engines:
            self.storage.flush_all()
        if 'mt5' in self.engines:
            self.mt5.shutdown()
        self.is_running = False


if __name__ == "__main__":
    orchestrator = PITSOrchestratorV2(dry_run=True)
    orchestrator.initialize_engines()
    orchestrator.run()
