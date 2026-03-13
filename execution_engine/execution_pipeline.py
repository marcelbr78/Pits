from .trade_executor import TradeExecutor
from .position_manager import PositionManager
from paper_trading.paper_trading_engine import PaperTradingEngine

class ExecutionPipeline:
    """
    Final decision layer. Applies filters and executes trades.
    """
    def __init__(self, executor: TradeExecutor, manager: PositionManager, paper_engine: PaperTradingEngine = None, live_trading: bool = False):
        self.logger = logging.getLogger("ExecutionPipeline")
        self.executor = executor
        self.manager = manager
        self.paper_engine = paper_engine
        self.live_trading = live_trading
        
        # Configuration thresholds
        self.prob_threshold = 0.75
        self.max_entropy = 0.55
        self.max_spread_pct = 0.0005 # 0.05% relative spread limit

    def process_signal(self, features: Dict[str, Any], prob_up: float, lot_size: float):
        """Processes signals and executes if all conditions are met."""
        symbol = features['symbol']
        entropy = features['entropy']
        spread = features['spread']
        mid_price = (features.get('bid', 0) + features.get('ask', 1)) / 2
        
        # 1. Check existing position
        if self.manager.has_position(symbol):
            return

        # 2. Apply Conditions
        passed_filters = True
        
        if prob_up < self.prob_threshold and prob_up > (1 - self.prob_threshold):
            passed_filters = False # Neutral probability
            
        if entropy > self.max_entropy:
            passed_filters = False # Too noisy
            
        if mid_price > 0 and (spread / mid_price) > self.max_spread_pct:
            passed_filters = False # Spread too high

        if not passed_filters:
            return

        # 3. Decision
        trade_type = "NONE"
        if prob_up >= self.prob_threshold:
            trade_type = "BUY"
        elif prob_up <= (1 - self.prob_threshold):
            trade_type = "SELL"

        if trade_type == "NONE":
            return

        # 4. Action
        if self.live_trading:
            self.logger.info(f"EXECUTION TRIGGERED: {trade_type} {symbol} | Prob: {prob_up:.2f} | Lot: {lot_size}")
            if trade_type == "BUY":
                self.executor.send_buy_order(symbol, lot_size)
            else:
                self.executor.send_sell_order(symbol, lot_size)
        elif self.paper_engine:
            self.logger.info(f"PAPER SIGNAL: {trade_type} {symbol} | Prob: {prob_up:.2f} | Lot: {lot_size}")
            self.paper_engine.open_trade(symbol, mid_price, lot_size, prob_up, trade_type)

    def update_tick(self, tick: Dict[str, Any]):
        """Passes real-time price updates to the paper trading engine."""
        if not self.live_trading and self.paper_engine:
            symbol = tick['symbol']
            mid_price = (tick['bid'] + tick['ask']) / 2
            self.paper_engine.update_trades(symbol, mid_price)
