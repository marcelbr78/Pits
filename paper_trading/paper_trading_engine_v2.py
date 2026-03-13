"""
Paper Trading Engine V2 — Fase 3.

Substitui SL/TP fixos (1%/2%) por valores dinâmicos baseados em:
  - ATR atual (realista para o mercado atual)
  - Padrão histórico matchado (Pattern Library)
  - Regime de volatilidade

SL/TP dinâmicos por regime:
  PANIC     : SL 2.5×ATR / TP 1.5×ATR (assimétrico — proteção)
  HIGH_VOL  : SL 2.0×ATR / TP 3.0×ATR
  NORMAL    : SL 1.5×ATR / TP 2.5×ATR
  LOW_VOL   : SL 1.0×ATR / TP 2.0×ATR
"""

import logging
import time
from typing import Dict, Any, Optional

from paper_trading.trade_logger import TradeLogger


# SL/TP multipliers por regime de volatilidade
REGIME_SLTP = {
    'PANIC':      {'sl_atr': 2.5, 'tp_atr': 1.5},
    'HIGH_VOL':   {'sl_atr': 2.0, 'tp_atr': 3.0},
    'NORMAL_VOL': {'sl_atr': 1.5, 'tp_atr': 2.5},
    'LOW_VOL':    {'sl_atr': 1.0, 'tp_atr': 2.0},
}

# Duração máxima de trade por regime
REGIME_DURATION = {
    'PANIC':      1800,   # 30 min
    'HIGH_VOL':   2700,   # 45 min
    'NORMAL_VOL': 3600,   # 60 min
    'LOW_VOL':    5400,   # 90 min
}


class PaperTradingEngineV2:
    """
    Paper Trading com SL/TP dinâmicos baseados em ATR e padrão.
    Drop-in replacement da PaperTradingEngine.
    """

    def __init__(self, trade_logger: TradeLogger):
        self.logger      = logging.getLogger("PaperTradingEngineV2")
        self.trade_logger = trade_logger
        self.open_trades: Dict[str, Dict[str, Any]] = {}

        # Fallback se ATR não disponível
        self._default_sl_pct = 0.01
        self._default_tp_pct = 0.02

    def open_trade(
        self,
        symbol: str,
        entry_price: float,
        lot_size: float,
        probability: float,
        side: str,
        atr: float = 0.0,
        vol_regime: str = 'NORMAL_VOL',
        pattern: Optional[Dict[str, Any]] = None,
    ):
        """Abre posição virtual com SL/TP dinâmicos."""
        if symbol in self.open_trades:
            return

        sl_dist, tp_dist = self._calc_sl_tp(
            entry_price, atr, vol_regime, pattern, side
        )

        if side == 'BUY':
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        max_dur = REGIME_DURATION.get(vol_regime, 3600)

        self.open_trades[symbol] = {
            'symbol':          symbol,
            'side':            side,
            'entry_price':     entry_price,
            'lot_size':        lot_size,
            'probability':     probability,
            'timestamp_entry': time.time(),
            'sl':              round(sl, 5),
            'tp':              round(tp, 5),
            'sl_dist':         round(sl_dist, 5),
            'tp_dist':         round(tp_dist, 5),
            'atr_at_entry':    round(atr, 5),
            'vol_regime':      vol_regime,
            'pattern':         pattern['name'] if pattern else 'N/A',
            'max_duration':    max_dur,
        }

        rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
        self.logger.info(
            f"PAPER OPEN V2: {side} {symbol} @ {entry_price} | "
            f"SL={sl:.4f} TP={tp:.4f} | ATR={atr:.4f} | "
            f"R:R={rr} | Regime={vol_regime}"
        )

    def update_trades(self, symbol: str, current_price: float):
        """Atualiza posição aberta com preço atual."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades[symbol]
        side  = trade['side']
        dur   = time.time() - trade['timestamp_entry']

        should_close, reason = False, ''

        if side == 'BUY':
            if current_price <= trade['sl']:
                should_close, reason = True, 'Stop Loss'
            elif current_price >= trade['tp']:
                should_close, reason = True, 'Take Profit'
        else:
            if current_price >= trade['sl']:
                should_close, reason = True, 'Stop Loss'
            elif current_price <= trade['tp']:
                should_close, reason = True, 'Take Profit'

        if dur >= trade['max_duration']:
            should_close, reason = True, 'Max Duration'

        if should_close:
            self.close_trade(symbol, current_price, reason)

    def close_trade(self, symbol: str, exit_price: float, reason: str = 'Manual'):
        if symbol not in self.open_trades:
            return

        trade = self.open_trades.pop(symbol)
        side  = trade['side']

        if side == 'BUY':
            pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price']
        else:
            pnl_pct = (trade['entry_price'] - exit_price) / trade['entry_price']

        # PnL em dólares (lote 0.01 WTI ≈ $10/pip → aproximado)
        pnl_usd = pnl_pct * trade['entry_price'] * trade['lot_size'] * 100

        completed = {
            **trade,
            'exit_price':    exit_price,
            'timestamp_exit': time.time(),
            'profit_loss':   round(pnl_pct, 6),
            'profit_usd':    round(pnl_usd, 4),
            'trade_duration': time.time() - trade['timestamp_entry'],
            'exit_reason':   reason,
        }

        self.trade_logger.log_trade(completed)
        self.logger.info(
            f"PAPER CLOSE V2: {symbol} @ {exit_price} | "
            f"{reason} | PnL={pnl_pct*100:.2f}% (${pnl_usd:.4f})"
        )

    def _calc_sl_tp(
        self,
        price: float,
        atr: float,
        vol_regime: str,
        pattern: Optional[Dict],
        side: str,
    ):
        """Calcula distâncias SL e TP dinamicamente."""
        mults = REGIME_SLTP.get(vol_regime, REGIME_SLTP['NORMAL_VOL'])
        sl_mult = mults['sl_atr']
        tp_mult = mults['tp_atr']

        # Ajuste por padrão histórico
        if pattern:
            sl_mult *= pattern.get('sl_multiplier', 1.0)
            tp_mult *= pattern.get('tp_multiplier', 1.0)

        if atr > 0:
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
        else:
            # Fallback percentual
            sl_dist = price * self._default_sl_pct * sl_mult
            tp_dist = price * self._default_tp_pct * tp_mult

        return sl_dist, tp_dist
