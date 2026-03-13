import threading
from typing import Dict, List, Any

class SystemState:
    """
    Thread-safe global state for the PITS trading system.
    Shared between the Orchestrator (writer) and the API (reader).
    """
    def __init__(self):
        self.lock = threading.Lock()
        
        # State fields
        self.signals: Dict[str, float] = {}       # {symbol: prob_up}
        self.features: Dict[str, Dict[str, Any]] = {}  # {symbol: {ofi, vwap, etc}}
        self.regime: Dict[str, str] = {
            "macro": "N/A",
            "volatility": "N/A",
            "micro": "N/A",
            "combined": "N/A"
        }
        self.positions: List[Dict[str, Any]] = []
        self.last_trades: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "profit_factor": 0.0
        }
        self.is_running: bool = False
        self.is_live: bool = False
        self.mt5_connected: bool = False
        self.next_event: Dict[str, Any] = {
            "name": "N/A",
            "minutes_remaining": 0,
            "impact": "N/A"
        }
        self.ticks_per_second: float = 0.0
        self.logs: List[str] = [] # Last 15 logs

    def update_signals(self, symbol: str, prob: float):
        with self.lock:
            self.signals[symbol] = prob

    def update_features(self, symbol: str, features: Dict[str, Any]):
        with self.lock:
            self.features[symbol] = features

    def update_regime(self, regime_data: Dict[str, str]):
        with self.lock:
            self.regime.update(regime_data)

    def update_positions(self, positions: List[Dict[str, Any]]):
        with self.lock:
            self.positions = positions

    def update_trades(self, trades: List[Dict[str, Any]]):
        with self.lock:
            self.last_trades = trades[-20:] # Keep last 20

    def update_metrics(self, metrics: Dict[str, Any]):
        with self.lock:
            self.metrics.update(metrics)

    def set_running(self, running: bool):
        with self.lock:
            self.is_running = running

    def set_live(self, live: bool):
        with self.lock:
            self.is_live = live

    def set_mt5(self, connected: bool):
        with self.lock:
            self.mt5_connected = connected

    def add_log(self, message: str):
        with self.lock:
            self.logs.append(message)
            if len(self.logs) > 15:
                self.logs.pop(0)

    def get_full_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "signals": self.signals,
                "features": self.features,
                "regime": self.regime,
                "positions": self.positions,
                "last_trades": self.last_trades,
                "metrics": self.metrics,
                "is_running": self.is_running,
                "is_live": self.is_live,
                "mt5_connected": self.mt5_connected,
                "next_event": self.next_event,
                "ticks_per_second": self.ticks_per_second,
                "logs": self.logs
            }
