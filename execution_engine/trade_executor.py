import MetaTrader5 as mt5
import logging
from typing import Dict, Any, Optional

class TradeExecutor:
    """
    Handles order execution via MetaTrader 5 API.
    """
    def __init__(self, magic_number: int = 123456, deviation: int = 20):
        self.logger = logging.getLogger("TradeExecutor")
        self.magic_number = magic_number
        self.deviation = deviation

    def _send_order(self, request: Dict[str, Any]) -> bool:
        """Internal helper to send order and log result."""
        result = mt5.order_send(request)
        if result is None:
            self.logger.error(f"Order failed, error code: {mt5.last_error()}")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed, retcode: {result.retcode} | {result.comment}")
            return False
            
        self.logger.info(f"Order executed successfully: {result.order} | {request['symbol']}")
        return True

    def send_buy_order(self, symbol: str, lot_size: float) -> bool:
        """Opens a BUY (long) position."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Could not get tick for {symbol}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "PITS Phase 1 Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return self._send_order(request)

    def send_sell_order(self, symbol: str, lot_size: float) -> bool:
        """Opens a SELL (short) position."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Could not get tick for {symbol}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "PITS Phase 1 Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return self._send_order(request)

    def close_position(self, symbol: str, ticket: int, order_type: int, volume: float) -> bool:
        """Closes a specific position by ticket."""
        tick = mt5.symbol_info_tick(symbol)
        close_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": ticket,
            "volume": volume,
            "type": close_type,
            "price": close_price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "PITS Phase 1 Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return self._send_order(request)
