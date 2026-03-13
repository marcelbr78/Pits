"""
Monte Carlo Simulator — Fase 4.

Roda 10.000 simulações para medir risco real.
Usado antes de qualquer mudança nos parâmetros.

Métricas:
  - P(ruína): probabilidade de zerar o capital
  - Drawdown esperado: percentil 95 do drawdown
  - Retorno esperado: percentil 50 (mediana)
  - Sharpe esperado: média das simulações
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.logger = logging.getLogger("MonteCarloSimulator")
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        trade_history: List[Dict[str, Any]],
        capital: float = 30.0,
        n_forward_trades: int = 100,
    ) -> Dict[str, Any]:
        """
        Roda n_simulations simulações com amostragem bootstrap.

        Args:
            trade_history: lista de trades passados com campo 'profit_loss'
            capital: capital inicial
            n_forward_trades: quantos trades simular no futuro

        Returns dict com métricas de risco.
        """
        if not trade_history:
            return self._empty_result()

        pnls = np.array([t.get('profit_loss', 0) for t in trade_history])
        if len(pnls) == 0:
            return self._empty_result()

        self.logger.info(
            f"Monte Carlo: {self.n_simulations} simulações × "
            f"{n_forward_trades} trades | base: {len(pnls)} trades históricos"
        )

        final_capitals = np.zeros(self.n_simulations)
        max_drawdowns  = np.zeros(self.n_simulations)
        ruin_count     = 0

        for i in range(self.n_simulations):
            # Bootstrap: sorteia sequência aleatória dos trades históricos
            sampled = self.rng.choice(pnls, size=n_forward_trades, replace=True)
            equity  = np.cumsum(sampled) + capital

            final_capitals[i] = equity[-1]

            # Drawdown máximo
            peak = np.maximum.accumulate(equity)
            dd   = (peak - equity) / peak
            max_drawdowns[i] = float(np.max(dd))

            if equity[-1] <= 0:
                ruin_count += 1

        p_ruin = ruin_count / self.n_simulations

        result = {
            'n_simulations':    self.n_simulations,
            'p_ruin':           round(p_ruin, 4),
            'p_ruin_pct':       round(p_ruin * 100, 2),
            'safe':             p_ruin < 0.005,  # < 0.5% de ruína = seguro
            'capital_median':   round(float(np.median(final_capitals)), 2),
            'capital_p10':      round(float(np.percentile(final_capitals, 10)), 2),
            'capital_p90':      round(float(np.percentile(final_capitals, 90)), 2),
            'drawdown_median':  round(float(np.median(max_drawdowns)), 4),
            'drawdown_p95':     round(float(np.percentile(max_drawdowns, 95)), 4),
            'expected_return_pct': round(
                (float(np.median(final_capitals)) - capital) / capital * 100, 2
            ),
        }

        self.logger.info(
            f"P(ruína)={result['p_ruin_pct']}% | "
            f"Capital mediana=${result['capital_median']} | "
            f"Drawdown p95={result['drawdown_p95']:.1%} | "
            f"{'SEGURO' if result['safe'] else 'RISCO ALTO'}"
        )

        return result

    def quick_check(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float = 30.0,
        n_trades: int = 100,
    ) -> Dict[str, Any]:
        """
        Versão rápida sem histórico real.
        Usa parâmetros estatísticos direto.
        """
        pnl_sim = np.where(
            self.rng.random((self.n_simulations, n_trades)) < win_rate,
            avg_win, -abs(avg_loss)
        )

        equities      = np.cumsum(pnl_sim, axis=1) + capital
        final_caps    = equities[:, -1]
        ruin_pct      = float(np.mean(final_caps <= 0)) * 100

        peaks         = np.maximum.accumulate(equities, axis=1)
        drawdowns     = np.max((peaks - equities) / peaks, axis=1)

        return {
            'p_ruin_pct':      round(ruin_pct, 2),
            'safe':            ruin_pct < 0.5,
            'capital_median':  round(float(np.median(final_caps)), 2),
            'drawdown_p95':    round(float(np.percentile(drawdowns, 95)), 4),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'n_simulations': 0, 'p_ruin': 0, 'p_ruin_pct': 0,
            'safe': True, 'capital_median': 0, 'capital_p10': 0,
            'capital_p90': 0, 'drawdown_median': 0, 'drawdown_p95': 0,
            'expected_return_pct': 0,
        }
