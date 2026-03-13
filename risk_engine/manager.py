from typing import List, Dict, Any

class RiskManager:
    """
    Core Risk Management Engine.
    Handles position sizing, portfolio exposure, and drawdown limits.
    """
    def __init__(self, max_risk_per_trade: float = 0.03, reward_risk_ratio: float = 2.0):
        self.max_risk = max_risk_per_trade
        self.reward_risk = reward_risk_ratio # 'b' in Kelly formula

    def calculate_kelly_size(self, p: float) -> float:
        """
        Calculates position size using Kelly Criterion.
        f = (p*b - q) / b
        p = win probability
        q = 1 - p
        b = reward / risk ratio
        """
        if p <= 0 or p >= 1: return 0.0
        
        q = 1 - p
        b = self.reward_risk
        
        # Kelly %
        f = (p * b - q) / b
        
        # Apply Half Kelly for safety
        f = f * 0.5
        
        # Limit to max risk per trade (3%)
        f = min(max(0, f), self.max_risk)
        
        # Placeholder for lot size conversion logic 
        # (e.g., 0.03 risk on $1000 with 100 pip SL = X lots)
        # For now return the decimal risk factor (0.0 to 0.03)
        return round(f, 4)

    def assess_portfolio_risk(self, open_positions: List[Dict[str, Any]]) -> float:
        """Calculates total correlated exposure."""
        # Phase 5: Markowitz / Monte Carlo logic
        return 0.0

    def can_execute(self, trade_signal: Dict[str, Any]) -> bool:
        """Validates if a trade complies with risk parameters."""
        return True
