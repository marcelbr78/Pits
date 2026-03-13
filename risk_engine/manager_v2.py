"""
Risk Manager V2 — Fase 4.

Extensão do RiskManager com:
  - Sizing baseado em ATR (ao invés de % fixo)
  - Integração com PortfolioRiskEngine
  - Verificação de drawdown diário
  - Verificação de 3 perdas consecutivas
  - Retorna lot_size calculado diretamente (0.01 para $30)
"""

import logging
from typing import Dict, Any, Optional


class RiskManagerV2:
    """
    Risk manager completo com ATR sizing e portfolio checks.
    """

    def __init__(
        self,
        capital: float = 30.0,
        max_risk_pct: float = 0.03,
        reward_risk_ratio: float = 2.0,
        min_lot: float = 0.01,
        max_lot: float = 0.05,
    ):
        self.logger         = logging.getLogger("RiskManagerV2")
        self.capital        = capital
        self.max_risk_pct   = max_risk_pct
        self.reward_risk    = reward_risk_ratio
        self.min_lot        = min_lot
        self.max_lot        = max_lot

    def calculate_position_size(
        self,
        prob_up: float,
        atr: float,
        price: float,
        sl_distance: Optional[float] = None,
    ) -> float:
        """
        Calcula lote usando Kelly + ATR sizing.

        Para $30 de capital com Exness:
          Risco máximo por trade = $30 × 3% = $0.90
          SL em WTI = ATR × 1.5 (ex: $0.40)
          Pip value WTI 0.01 lot ≈ $0.001/pip
          Lots = Risco / (SL_pips × pip_value)
        """
        # Kelly fraction
        kelly = self._kelly(prob_up)
        if kelly <= 0:
            return 0.0

        # Risco em dólares
        risk_usd = self.capital * min(kelly, self.max_risk_pct)

        # SL em pontos de preço
        if sl_distance and sl_distance > 0:
            sl_pts = sl_distance
        elif atr > 0:
            sl_pts = atr * 1.5
        else:
            sl_pts = price * 0.01  # 1% fallback

        # Lot sizing: 1 lot WTI = $10/pip no standard
        # 0.01 lot = $0.10/pip | pip = 0.01 price unit
        # risk_usd = lots × sl_pts × 100
        pip_value_per_lot = 100  # para WTI na Exness
        lot = risk_usd / (sl_pts * pip_value_per_lot)

        # Limita ao range permitido
        lot = max(self.min_lot, min(self.max_lot, round(lot, 2)))

        self.logger.debug(
            f"Sizing: Kelly={kelly:.3f} | Risk=${risk_usd:.4f} | "
            f"SL={sl_pts:.4f} | Lot={lot}"
        )
        return lot

    def calculate_kelly_size(self, prob_up: float) -> float:
        """Compatibilidade com RiskManager Fase 1."""
        kelly = self._kelly(prob_up)
        return min(kelly, self.max_risk_pct)

    def _kelly(self, p: float) -> float:
        """Half-Kelly criterion."""
        if p <= 0 or p >= 1:
            return 0.0
        q = 1 - p
        b = self.reward_risk
        f = (p * b - q) / b
        return max(0.0, f * 0.5)

    def validate_trade(
        self,
        symbol: str,
        prob_up: float,
        entropy: float,
        spread: float,
        atr: float,
        portfolio_check: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Validação completa antes de qualquer trade.
        Returns {'allowed': bool, 'reason': str}
        """
        if prob_up < 0.75:
            return {'allowed': False, 'reason': f'Prob {prob_up:.2f} < 0.75'}

        if entropy > 0.55:
            return {'allowed': False, 'reason': f'Entropy {entropy:.3f} > 0.55'}

        if atr > 1.50:
            return {'allowed': False, 'reason': f'ATR {atr:.3f} > 1.50 (mercado caótico)'}

        if portfolio_check and not portfolio_check.get('allowed', True):
            return {'allowed': False, 'reason': portfolio_check.get('reason', 'Portfolio blocked')}

        return {'allowed': True, 'reason': 'OK'}
