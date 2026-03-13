import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

class PerformanceTracker:
    """
    Calculates trading performance metrics from simulated trades.
    """
    def __init__(self):
        self.logger = logging.getLogger("PerformanceTracker")

    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates Key Performance Indicators (KPIs).
        """
        if trades_df.empty:
            return {
                "win_rate": 0.0,
                "average_profit": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }

        pnl = trades_df['profit_loss'].values
        
        # Win Rate
        wins = pnl[pnl > 0]
        win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
        
        # Average Profit
        avg_profit = np.mean(pnl)
        
        # Max Drawdown
        cumulative_pnl = np.cumsum(pnl)
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Sharpe Ratio (Simplified from trade PnL series)
        std_pnl = np.std(pnl)
        sharpe = (avg_profit / std_pnl) * np.sqrt(252) if std_pnl != 0 else 0.0

        metrics = {
            "win_rate": round(win_rate, 4),
            "average_profit": round(avg_profit, 6),
            "max_drawdown": round(max_dd, 6),
            "sharpe_ratio": round(sharpe, 4),
            "total_trades": len(pnl)
        }
        
        self.logger.info(f"Performance Metrics: {metrics}")
        return metrics
    def get_summary(self, path: str = "data/paper_trades.parquet") -> Dict[str, float]:
        """
        Loads trades from disk and returns current metrics.
        """
        import os
        if not os.path.exists(path):
            return self.calculate_metrics(pd.DataFrame())
            
        try:
            df = pd.read_parquet(path)
            return self.calculate_metrics(df)
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
            return self.calculate_metrics(pd.DataFrame())
