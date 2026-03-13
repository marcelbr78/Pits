import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any

class PerformanceAnalyzer:
    """
    Analyzes historical paper trading data to calculate performance KPIs.
    """
    def __init__(self, data_path: str = "data/paper_trades.parquet"):
        self.data_path = data_path
        self.logger = logging.getLogger("PerformanceAnalyzer")

    def get_summary(self) -> Dict[str, Any]:
        """Reads paper trades and returns a performance summary."""
        if not os.path.exists(self.data_path):
            self.logger.warning(f"No paper trades found at {self.data_path}")
            return {}

        try:
            df = pd.read_parquet(self.data_path)
            if df.empty:
                return {}

            pnl = df['profit_loss'].values
            
            # Win Rate
            win_rate = len(pnl[pnl > 0]) / len(pnl)
            
            # Average Return
            avg_return = np.mean(pnl)
            
            # Profit Factor
            gross_profits = np.sum(pnl[pnl > 0])
            gross_losses = np.abs(np.sum(pnl[pnl < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Max Drawdown
            cum_pnl = np.cumsum(pnl)
            peak = np.maximum.accumulate(cum_pnl)
            drawdown = peak - cum_pnl
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            # Sharpe Ratio
            std_pnl = np.std(pnl)
            sharpe = (avg_return / std_pnl) * np.sqrt(252) if std_pnl != 0 else 0.0

            summary = {
                "win_rate": round(win_rate, 4),
                "average_return": round(avg_return, 6),
                "profit_factor": round(profit_factor, 2),
                "max_drawdown": round(max_dd, 6),
                "sharpe_ratio": round(sharpe, 4),
                "total_trades": len(df)
            }
            
            self.logger.info(f"Performance Summary: {summary}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {}
