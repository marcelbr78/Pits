"""
Ensemble Model V2 — Fase 5.

Combina todos os modelos das Fases 1-5:
  30% XGBoost    — padrões não-lineares (Fase 1-4)
  20% Bayes      — atualização rápida (Fase 1)
  25% LSTM       — padrões temporais (Fase 5)
  25% GNN        — correlação multi-ativo (Fase 5)

Pesos ajustados dinamicamente por regime:
  WAR regime     → GNN domina (correlações importam mais)
  NEWS regime    → Bayes domina (reage mais rápido)
  NORMAL regime  → XGBoost domina (padrões históricos)
"""

import logging
from typing import Dict, Any, Optional


# Pesos por regime
REGIME_WEIGHTS = {
    'WAR': {
        'xgb': 0.20, 'bayes': 0.15, 'lstm': 0.25, 'gnn': 0.40
    },
    'RISK_OFF': {
        'xgb': 0.25, 'bayes': 0.20, 'lstm': 0.30, 'gnn': 0.25
    },
    'NEWS_DRIVEN': {
        'xgb': 0.20, 'bayes': 0.40, 'lstm': 0.20, 'gnn': 0.20
    },
    'CRISIS': {
        'xgb': 0.10, 'bayes': 0.20, 'lstm': 0.30, 'gnn': 0.40
    },
    'DEFAULT': {
        'xgb': 0.30, 'bayes': 0.20, 'lstm': 0.25, 'gnn': 0.25
    },
}


class EnsembleV2:
    """
    Ensemble dinâmico de todos os modelos.
    Fallback gracioso — se modelo não disponível, redistribui peso.
    """

    def __init__(self):
        self.logger = logging.getLogger("EnsembleV2")

    def predict(
        self,
        xgb_prob:   float,
        bayes_prob: float,
        lstm_prob:  Optional[float] = None,
        gnn_prob:   Optional[float] = None,
        macro_regime: str = 'DEFAULT',
        micro_regime: str = 'TRENDING',
    ) -> Dict[str, Any]:
        """
        Calcula probabilidade final com pesos dinâmicos.
        
        Returns dict com:
          probability: float — probabilidade final
          weights_used: dict — pesos aplicados
          confidence: float — dispersão entre modelos (baixa = alta confiança)
          breakdown: dict — contribuição de cada modelo
        """
        # Seleciona pesos por regime
        regime_key = macro_regime if macro_regime in REGIME_WEIGHTS else 'DEFAULT'
        if micro_regime == 'NEWS_DRIVEN':
            regime_key = 'NEWS_DRIVEN'
        weights = dict(REGIME_WEIGHTS[regime_key])

        # Fallback se modelos não disponíveis
        probs: Dict[str, Optional[float]] = {
            'xgb':   xgb_prob,
            'bayes': bayes_prob,
            'lstm':  lstm_prob,
            'gnn':   gnn_prob,
        }

        available = {k: v for k, v in probs.items() if v is not None and 0 <= v <= 1}
        if not available:
            return {'probability': 0.5, 'weights_used': {}, 'confidence': 0.0, 'breakdown': {}}

        # Redistribui pesos dos modelos indisponíveis
        unavailable_weight = sum(weights[k] for k in weights if k not in available)
        if unavailable_weight > 0 and available:
            scale = 1.0 / (1.0 - unavailable_weight)
            adjusted_weights = {k: weights[k] * scale for k in available}
        else:
            adjusted_weights = {k: weights[k] for k in available}

        # Calcula probabilidade final
        final_prob = sum(adjusted_weights[k] * available[k] for k in available)

        # Confiança = 1 - dispersão entre modelos
        vals = list(available.values())
        if len(vals) > 1:
            import numpy as np
            dispersion = float(np.std(vals))
            confidence = round(max(0.0, 1.0 - dispersion * 2), 4)
        else:
            confidence = 0.5

        breakdown = {k: round(adjusted_weights[k] * available[k], 4) for k in available}

        return {
            'probability':   round(final_prob, 4),
            'weights_used':  adjusted_weights,
            'confidence':    confidence,
            'breakdown':     breakdown,
            'regime_used':   regime_key,
            'models_active': len(available),
        }
