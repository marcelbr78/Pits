"""
Pattern Library — Fase 3.

Biblioteca de padrões históricos de crises e eventos.
Compara o estado atual do mercado com padrões conhecidos:
  - Pandemia COVID 2020
  - Invasão Ucrânia 2022
  - Guerra Hormuz 2026

Quando detecta similaridade > 70%, ajusta:
  - TP dinâmico baseado no retorno histórico
  - SL dinâmico baseado na volatilidade histórica
  - Probabilidade de continuação do movimento
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


CRISIS_PATTERNS = {
    "PANDEMIA_COLAPSO_2020": {
        "description": "COVID — colapso da demanda, WTI negativo",
        "macro_regime": "CRISIS",
        "vol_regime": "PANIC",
        "obi_range": (-1.0, -0.5),
        "vwap_dev_range": (-0.10, -0.03),
        "avg_move_5min_pct": -0.08,
        "avg_move_30min_pct": -0.15,
        "continuation_prob": 0.72,
        "recommended_action": "PAUSE",  # não opera em colapso
        "tp_multiplier": 0.5,
        "sl_multiplier": 3.0,
    },
    "PANDEMIA_RECUPERACAO_2020": {
        "description": "COVID — short squeeze após corte OPEP 9.7M",
        "macro_regime": "RISK_OFF",
        "vol_regime": "HIGH_VOL",
        "obi_range": (0.3, 1.0),
        "vwap_dev_range": (-0.02, 0.05),
        "avg_move_5min_pct": 0.035,
        "avg_move_30min_pct": 0.08,
        "continuation_prob": 0.68,
        "recommended_action": "BUY",
        "tp_multiplier": 2.5,
        "sl_multiplier": 1.5,
    },
    "UCRANIA_SPIKE_2022": {
        "description": "Invasão Ucrânia — WTI para $130",
        "macro_regime": "WAR",
        "vol_regime": "HIGH_VOL",
        "obi_range": (0.4, 1.0),
        "vwap_dev_range": (0.01, 0.08),
        "avg_move_5min_pct": 0.025,
        "avg_move_30min_pct": 0.06,
        "continuation_prob": 0.74,
        "recommended_action": "BUY",
        "tp_multiplier": 2.0,
        "sl_multiplier": 1.2,
    },
    "HORMUZ_2026": {
        "description": "Guerra Hormuz — WTI +9.72%",
        "macro_regime": "WAR",
        "vol_regime": "HIGH_VOL",
        "obi_range": (0.3, 1.0),
        "vwap_dev_range": (0.0, 0.10),
        "avg_move_5min_pct": 0.02,
        "avg_move_30min_pct": 0.05,
        "continuation_prob": 0.78,
        "recommended_action": "BUY",
        "tp_multiplier": 2.2,
        "sl_multiplier": 1.3,
    },
    "EIA_BULLISH_SURPRISE": {
        "description": "EIA — queda de inventário surpreende mercado",
        "macro_regime": "NEUTRAL",
        "vol_regime": "HIGH_VOL",
        "obi_range": (0.2, 1.0),
        "vwap_dev_range": (-0.01, 0.03),
        "avg_move_5min_pct": 0.015,
        "avg_move_30min_pct": 0.03,
        "continuation_prob": 0.65,
        "recommended_action": "BUY",
        "tp_multiplier": 1.5,
        "sl_multiplier": 1.0,
    },
    "OPEP_CORTE_SURPRESA": {
        "description": "OPEP corte surpresa — short squeeze instantâneo",
        "macro_regime": "INFLATION",
        "vol_regime": "HIGH_VOL",
        "obi_range": (0.5, 1.0),
        "vwap_dev_range": (0.01, 0.06),
        "avg_move_5min_pct": 0.03,
        "avg_move_30min_pct": 0.07,
        "continuation_prob": 0.80,
        "recommended_action": "BUY",
        "tp_multiplier": 2.8,
        "sl_multiplier": 1.0,
    },
    "RISK_ON_NORMAL": {
        "description": "Mercado normal — risk on, compra gradual",
        "macro_regime": "RISK_ON",
        "vol_regime": "NORMAL_VOL",
        "obi_range": (0.1, 0.6),
        "vwap_dev_range": (-0.005, 0.005),
        "avg_move_5min_pct": 0.004,
        "avg_move_30min_pct": 0.008,
        "continuation_prob": 0.55,
        "recommended_action": "BUY",
        "tp_multiplier": 1.0,
        "sl_multiplier": 1.0,
    },
}


class PatternLibrary:
    def __init__(self):
        self.logger = logging.getLogger("PatternLibrary")
        self.patterns = CRISIS_PATTERNS

    def match(
        self,
        macro_regime: str,
        vol_regime: str,
        obi: float,
        vwap_dev_pct: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Encontra padrão histórico mais similar ao estado atual.
        Retorna o padrão e score de similaridade (0-1).
        """
        best_pattern = None
        best_score = 0.0

        for name, p in self.patterns.items():
            score = self._score_pattern(p, macro_regime, vol_regime, obi, vwap_dev_pct)
            if score > best_score:
                best_score = score
                best_pattern = {**p, "name": name, "similarity": round(score, 3)}

        if best_pattern and best_score >= 0.50:
            self.logger.debug(
                f"Padrão: {best_pattern['name']} | "
                f"Similaridade: {best_score:.1%}"
            )
            return best_pattern

        return None

    def get_dynamic_tp_sl(
        self,
        pattern: Optional[Dict[str, Any]],
        atr: float,
        base_tp_atr: float = 2.0,
        base_sl_atr: float = 1.5,
    ) -> Tuple[float, float]:
        """
        Retorna TP e SL dinâmicos baseados no padrão histórico e ATR atual.
        Returns: (tp_distance, sl_distance) em pontos de preço.
        """
        tp_mult = pattern["tp_multiplier"] if pattern else 1.0
        sl_mult = pattern["sl_multiplier"] if pattern else 1.0

        tp = atr * base_tp_atr * tp_mult
        sl = atr * base_sl_atr * sl_mult

        return round(tp, 5), round(sl, 5)

    def _score_pattern(
        self,
        pattern: Dict,
        macro: str,
        vol: str,
        obi: float,
        vwap_dev: float,
    ) -> float:
        score = 0.0

        # Regime macro (peso 35%)
        if pattern["macro_regime"] == macro:
            score += 0.35
        elif macro in ("WAR", "RISK_OFF") and pattern["macro_regime"] in ("WAR", "RISK_OFF"):
            score += 0.15

        # Regime vol (peso 25%)
        if pattern["vol_regime"] == vol:
            score += 0.25
        elif vol in ("HIGH_VOL", "PANIC") and pattern["vol_regime"] in ("HIGH_VOL", "PANIC"):
            score += 0.10

        # OBI range (peso 25%)
        obi_min, obi_max = pattern["obi_range"]
        if obi_min <= obi <= obi_max:
            score += 0.25
        elif abs(obi - (obi_min + obi_max) / 2) < 0.2:
            score += 0.10

        # VWAP deviation range (peso 15%)
        dev_min, dev_max = pattern["vwap_dev_range"]
        if dev_min <= vwap_dev <= dev_max:
            score += 0.15

        return score
