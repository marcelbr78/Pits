import logging
import time
from typing import Dict, Any, List
from .trade_logger import TradeLogger

class PaperTradingEngine:
    """
    Simulates a live trading environment with virtual positions and SL/TP logic.
    """
    def __init__(self, logger: TradeLogger):
        self.logger = logging.getLogger("PaperTradingEngine")
        self.trade_logger = logger
        self.open_trades: Dict[str, Dict[str, Any]] = {}
        
        # Simulation parameters
        self.stop_loss_pct = 0.01   # 1% Fixed SL
        self.take_profit_pct = 0.02 # 2% Fixed TP
        self.max_duration_sec = 3600 # 1 hour max duration

    def open_trade(self, symbol: str, entry_price: float, lot_size: float, probability: float, side: str):
        """Opens a virtual position."""
        if symbol in self.open_trades:
            self.logger.debug(f"Position already open for {symbol} in paper engine.")
            return

        self.open_trades[symbol] = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'lot_size': lot_size,
            'probability': probability,
            'timestamp_entry': time.time(),
            'sl': entry_price * (1 - self.stop_loss_pct) if side == "BUY" else entry_price * (1 + self.stop_loss_pct),
            'tp': entry_price * (1 + self.take_profit_pct) if side == "BUY" else entry_price * (1 - self.take_profit_pct)
        }
        self.logger.info(f"PAPER OPEN: {side} {symbol} @ {entry_price} | Prob: {probability:.2f}")

    def update_trades(self, symbol: str, current_price: float):
        """Updates active trades with current price and checks for SL/TP/Expiry."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades[symbol]
        side = trade['side']
        duration = time.time() - trade['timestamp_entry']
        
        should_close = False
        reason = ""

        # Check SL/TP
        if side == "BUY":
            if current_price <= trade['sl']:
                should_close, reason = True, "Stop Loss"
            elif current_price >= trade['tp']:
                should_close, reason = True, "Take Profit"
        else: # SELL
            if current_price >= trade['sl']:
                should_close, reason = True, "Stop Loss"
            elif current_price <= trade['tp']:
                should_close, reason = True, "Take Profit"

        # Check Duration
        if duration >= self.max_duration_sec:
            should_close, reason = True, "Max Duration Exceeded"

        if should_close:
            self.logger.info(f"PAPER CLOSE: {symbol} @ {current_price} | Reason: {reason}")
            self.close_trade(symbol, current_price, reason)

    def close_trade(self, symbol: str, exit_price: float, reason: str = "Manual"):
        """Closes a virtual position and logs result."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades.pop(symbol)
        side = trade['side']
        
        # Calculate PnL (in percentage for simplicity in Phase 1)
        if side == "BUY":
            pnl = (exit_price - trade['entry_price']) / trade['entry_price']
        else:
            pnl = (trade['entry_price'] - exit_price) / trade['entry_price']

        completed_trade = {
            **trade,
            'exit_price': exit_price,
            'timestamp_exit': time.time(),
            'profit_loss': pnl,
            'trade_duration': time.time() - trade['timestamp_entry'],
            'exit_reason': reason
        }
        
        self.trade_logger.log_trade(completed_trade)
