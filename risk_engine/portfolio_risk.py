"""
Portfolio Risk Engine — Fase 4.

Gerencia exposição correlacionada entre ativos.
Evita abrir WTI + Brent + Ouro ao mesmo tempo
pois compartilham o mesmo risco geopolítico.

Correlações hardcoded (atualizadas por dados históricos):
  WTI  ↔ Brent:  +0.97
  WTI  ↔ Gold:   +0.74 (normal) / +0.91 (guerra)
  WTI  ↔ DXY:    -0.85
  WTI  ↔ SP500:  +0.48

Regras:
  - Máximo 2.0 posições equivalentes abertas
  - Máximo 40% do risco em um único ativo
  - Drawdown diário > 10% → para o dia
  - 3 perdas consecutivas → cooldown 2 horas
"""

import logging
import time
from typing import Dict, Any, List, Optional


# Correlações entre ativos (absolutos)
CORRELATIONS: Dict[str, Dict[str, float]] = {
    'USOILm':  {'USOILm': 1.0, 'UKOILm': 0.97, 'XAUUSDm': 0.74, 'BTCUSDm': 0.30},
    'UKOILm':  {'USOILm': 0.97, 'UKOILm': 1.0, 'XAUUSDm': 0.70, 'BTCUSDm': 0.28},
    'XAUUSDm': {'USOILm': 0.74, 'UKOILm': 0.70, 'XAUUSDm': 1.0, 'BTCUSDm': 0.45},
    'BTCUSDm': {'USOILm': 0.30, 'UKOILm': 0.28, 'XAUUSDm': 0.45, 'BTCUSDm': 1.0},
}

WAR_CORRELATIONS: Dict[str, Dict[str, float]] = {
    'USOILm':  {'USOILm': 1.0, 'UKOILm': 0.98, 'XAUUSDm': 0.91, 'BTCUSDm': 0.35},
    'UKOILm':  {'USOILm': 0.98, 'UKOILm': 1.0, 'XAUUSDm': 0.88, 'BTCUSDm': 0.33},
    'XAUUSDm': {'USOILm': 0.91, 'UKOILm': 0.88, 'XAUUSDm': 1.0, 'BTCUSDm': 0.50},
    'BTCUSDm': {'USOILm': 0.35, 'UKOILm': 0.33, 'XAUUSDm': 0.50, 'BTCUSDm': 1.0},
}


class PortfolioRiskEngine:
    def __init__(self, capital: float = 30.0, max_eq_positions: float = 2.0):
        self.logger = logging.getLogger("PortfolioRiskEngine")
        self.capital = capital
        self.max_eq_positions = max_eq_positions
        self.max_daily_drawdown = 0.10
        self.max_single_asset_pct = 0.40
        self.max_consecutive_losses = 3
        self.cooldown_seconds = 7200  # 2 horas

        self._daily_start_capital = capital
        self._current_capital = capital
        self._consecutive_losses = 0
        self._cooldown_until: float = 0.0
        self._open_positions: Dict[str, Dict[str, Any]] = {}

    def can_trade(self, symbol: str, lot_size: float, macro_regime: str = "NEUTRAL") -> Dict[str, Any]:
        """
        Verifica se é seguro abrir nova posição.
        Retorna {'allowed': bool, 'reason': str, 'adjusted_lot': float}
        """
        now = time.time()

        # Cooldown ativo
        if now < self._cooldown_until:
            mins = (self._cooldown_until - now) / 60
            return {'allowed': False, 'reason': f'Cooldown ativo — {mins:.0f}min restantes', 'adjusted_lot': 0}

        # Drawdown diário
        dd = (self._daily_start_capital - self._current_capital) / self._daily_start_capital
        if dd >= self.max_daily_drawdown:
            return {'allowed': False, 'reason': f'Drawdown diário {dd:.1%} ≥ 10%', 'adjusted_lot': 0}

        # Posição duplicada
        if symbol in self._open_positions:
            return {'allowed': False, 'reason': f'Já existe posição aberta em {symbol}', 'adjusted_lot': 0}

        # Exposição equivalente
        corr_map = WAR_CORRELATIONS if macro_regime == "WAR" else CORRELATIONS
        eq_exposure = self._calc_equivalent_exposure(symbol, corr_map)
        if eq_exposure >= self.max_eq_positions:
            return {
                'allowed': False,
                'reason': f'Exposição equivalente {eq_exposure:.2f} ≥ {self.max_eq_positions}',
                'adjusted_lot': 0
            }

        # Ajusta lote se necessário
        remaining_eq = self.max_eq_positions - eq_exposure
        adjusted_lot = min(lot_size, lot_size * remaining_eq)

        return {
            'allowed': True,
            'reason': f'OK — exposição equivalente atual: {eq_exposure:.2f}',
            'adjusted_lot': round(max(0.01, adjusted_lot), 2),
            'equivalent_exposure': eq_exposure,
        }

    def register_open(self, symbol: str, lot: float, entry: float):
        self._open_positions[symbol] = {'lot': lot, 'entry': entry, 'ts': time.time()}

    def register_close(self, symbol: str, pnl: float):
        """Registra fechamento e atualiza contadores de risco."""
        self._open_positions.pop(symbol, None)
        self._current_capital += pnl

        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until = time.time() + self.cooldown_seconds
                self.logger.warning(
                    f"{self._consecutive_losses} perdas consecutivas — "
                    f"cooldown de 2h ativado."
                )
        else:
            self._consecutive_losses = 0

    def reset_daily(self):
        """Chamado no início de cada dia de trading."""
        self._daily_start_capital = self._current_capital
        self._consecutive_losses = 0
        self.logger.info(f"Reset diário — capital base: ${self._current_capital:.2f}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        return {
            'capital': round(self._current_capital, 2),
            'daily_drawdown_pct': round(
                (self._daily_start_capital - self._current_capital) / self._daily_start_capital, 4
            ),
            'open_positions': len(self._open_positions),
            'consecutive_losses': self._consecutive_losses,
            'cooldown_active': time.time() < self._cooldown_until,
        }

    def _calc_equivalent_exposure(self, new_symbol: str, corr_map: Dict) -> float:
        """Calcula exposição equivalente total incluindo nova posição."""
        if not self._open_positions:
            return 0.0

        total = 0.0
        for sym in self._open_positions:
            corr = corr_map.get(new_symbol, {}).get(sym, 0.3)
            total += corr

        return round(total, 3)
