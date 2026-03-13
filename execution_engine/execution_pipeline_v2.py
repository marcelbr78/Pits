"""
Execution Pipeline V2 — Fase 4.

Integra:
  - RiskManagerV2 (ATR sizing)
  - PortfolioRiskEngine (correlações, drawdown)
  - PaperTradingEngineV2 (SL/TP dinâmicos)
  - Threshold dinâmico por calendário e regime

Mantém API compatível com ExecutionPipeline Fase 1.
"""

import logging
from typing import Dict, Any, Optional

from execution_engine.trade_executor import TradeExecutor
from execution_engine.position_manager import PositionManager
from paper_trading.paper_trading_engine_v2 import PaperTradingEngineV2
from risk_engine.manager_v2 import RiskManagerV2
from risk_engine.portfolio_risk import PortfolioRiskEngine


class ExecutionPipelineV2:
    """
    Pipeline de execução completa — Fase 4.
    """

    def __init__(
        self,
        executor: TradeExecutor,
        position_manager: PositionManager,
        paper_engine: PaperTradingEngineV2,
        risk_manager: RiskManagerV2,
        portfolio_risk: PortfolioRiskEngine,
        live_trading: bool = False,
    ):
        self.logger          = logging.getLogger("ExecutionPipelineV2")
        self.executor        = executor
        self.manager         = position_manager
        self.paper_engine    = paper_engine
        self.risk_manager    = risk_manager
        self.portfolio_risk  = portfolio_risk
        self.live_trading    = live_trading

        # Limites
        self.max_entropy    = 0.55
        self.max_spread_rel = 0.0005

    def process_signal(
        self,
        features: Dict[str, Any],
        prob_up: float,
        lot_size: float,
        prob_threshold: float = 0.75,
        atr: float = 0.0,
        vol_regime: str = 'NORMAL_VOL',
        pattern: Optional[Dict] = None,
        macro_regime: str = 'NEUTRAL',
        sl_distance: Optional[float] = None,
    ):
        """
        Processa sinal com todos os filtros de segurança.
        Só executa se TODOS os critérios forem atendidos.
        """
        symbol  = features['symbol']
        entropy = features.get('entropy', 1.0)
        spread  = features.get('spread', 999)
        mid     = (features.get('bid', 0) + features.get('ask', 1)) / 2

        # 1. Posição duplicada
        if self.manager.has_position(symbol):
            return

        # 2. Probabilidade
        if prob_up < prob_threshold:
            return

        # 3. Apenas BUY (regra do sistema)
        if prob_up < prob_threshold:
            return

        # 4. Entropy
        if entropy > self.max_entropy:
            return

        # 5. Spread
        if mid > 0 and (spread / mid) > self.max_spread_rel:
            return

        # 6. ATR
        if atr > 1.50:
            return

        # 7. Portfolio risk check
        port_check = self.portfolio_risk.can_trade(symbol, lot_size, macro_regime)
        if not port_check['allowed']:
            self.logger.debug(f"Portfolio bloqueou {symbol}: {port_check['reason']}")
            return

        # Sizing final pelo risk manager V2
        final_lot = self.risk_manager.calculate_position_size(
            prob_up, atr, mid, sl_distance
        )
        final_lot = max(0.01, final_lot)

        # 8. Executa
        if self.live_trading:
            self.logger.info(
                f"LIVE BUY: {symbol} @ {mid:.4f} | "
                f"Lot={final_lot} | Prob={prob_up:.2f} | ATR={atr:.4f}"
            )
            self.executor.send_buy_order(symbol, final_lot)
            self.portfolio_risk.register_open(symbol, final_lot, mid)
        else:
            self.paper_engine.open_trade(
                symbol, mid, final_lot, prob_up,
                side='BUY',
                atr=atr,
                vol_regime=vol_regime,
                pattern=pattern,
            )
            self.logger.info(
                f"PAPER BUY: {symbol} @ {mid:.4f} | "
                f"Lot={final_lot} | Prob={prob_up:.2f} | Regime={vol_regime}"
            )

    def update_tick(self, tick: Dict[str, Any]):
        """Repassa tick para o paper engine atualizar SL/TP."""
        if not self.live_trading:
            symbol = tick['symbol']
            mid    = (tick['bid'] + tick['ask']) / 2
            self.paper_engine.update_trades(symbol, mid)
