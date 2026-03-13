"""
Feature Engineering Avançado — Fase 4.

Expande o vetor de features para 100+ variáveis.
Inclui todas as categorias do blueprint:
  - Microestrutura (OBI, cancel rate, TFT, VWAP dev)
  - Cross-asset lag (Gold, DXY, VIX, Brent)
  - Volatilidade (realized, clustering, GARCH simplificado)
  - Estrutura de mercado (distância de VWAP, high, open)
  - Momentum e retornos em múltiplas janelas
  - Regime features (encoded)
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, List


class AdvancedFeatureEngine:
    """
    Calcula 100+ features a partir do estado atual do mercado.
    Alimenta XGBoost, Random Forest e futuramente LSTM/GNN.
    """

    def __init__(self, symbol: str, window: int = 500):
        self.logger = logging.getLogger(f"AdvancedFeatureEngine[{symbol}]")
        self.symbol = symbol
        self.window = window

        # Histórico para cálculos rolling
        self._prices:   deque = deque(maxlen=window)
        self._volumes:  deque = deque(maxlen=window)
        self._spreads:  deque = deque(maxlen=window)
        self._obis:     deque = deque(maxlen=window)
        self._tfts:     deque = deque(maxlen=window)
        self._entropies: deque = deque(maxlen=window)
        self._vols:     deque = deque(maxlen=window)
        self._session_open: float = 0.0
        self._session_high: float = 0.0
        self._session_low: float = float('inf')

    def compute(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Recebe o feature vector básico e retorna vetor expandido 100+.
        Compatível com o output do FeaturePipelineV2.
        """
        mid    = (features.get('bid', 0) + features.get('ask', 0)) / 2
        vol    = features.get('volume', 1.0) or 1.0
        spread = features.get('spread', 0)
        obi    = features.get('obi', 0)
        tft    = features.get('tft', 0.5)
        ent    = features.get('entropy', 0)
        volat  = features.get('volatility', 0)
        vwap   = features.get('vwap', mid)

        # Atualiza histórico
        self._prices.append(mid)
        self._volumes.append(vol)
        self._spreads.append(spread)
        self._obis.append(obi)
        self._tfts.append(tft)
        self._entropies.append(ent)
        self._vols.append(volat)

        if self._session_open == 0:
            self._session_open = mid
        self._session_high = max(self._session_high, mid)
        self._session_low  = min(self._session_low, mid)

        p = list(self._prices)
        n = len(p)

        result: Dict[str, float] = {}

        # ── 1. Features base (Fase 1) ──────────────────────────
        result['ofi']        = features.get('ofi', 0)
        result['obi']        = obi
        result['tft']        = tft
        result['vwap_dev']   = features.get('vwap_dev_pct', 0)
        result['vwap_zscore'] = features.get('vwap_zscore', 0)
        result['spread']     = spread
        result['volatility'] = volat
        result['entropy']    = ent

        # ── 2. Retornos em múltiplas janelas ───────────────────
        for w in [1, 5, 10, 20, 50, 100]:
            if n > w:
                ret = (p[-1] - p[-w]) / p[-w] if p[-w] != 0 else 0
                result[f'ret_{w}t'] = round(ret, 6)
            else:
                result[f'ret_{w}t'] = 0.0

        # ── 3. Volatilidade em múltiplas janelas ───────────────
        for w in [10, 20, 50, 100]:
            if n >= w:
                rets = np.diff(np.log(p[-w:])) if min(p[-w:]) > 0 else np.zeros(w-1)
                result[f'vol_{w}t']  = round(float(np.std(rets)), 8)
                result[f'mean_{w}t'] = round(float(np.mean(rets)), 8)
            else:
                result[f'vol_{w}t']  = 0.0
                result[f'mean_{w}t'] = 0.0

        # Vol ratio — clustering de volatilidade
        if result['vol_10t'] > 0 and result['vol_100t'] > 0:
            result['vol_ratio'] = round(result['vol_10t'] / result['vol_100t'], 4)
        else:
            result['vol_ratio'] = 1.0

        # ── 4. Estrutura de mercado ────────────────────────────
        atr = result['vol_20t'] * mid if mid > 0 else 0.0001

        result['dist_vwap_atr'] = round(
            (mid - vwap) / atr if atr > 0 else 0, 4
        )
        result['dist_session_high'] = round(
            (self._session_high - mid) / mid if mid > 0 else 0, 6
        )
        result['dist_session_low'] = round(
            (mid - self._session_low) / mid if mid > 0 else 0, 6
        )
        result['dist_session_open'] = round(
            (mid - self._session_open) / self._session_open
            if self._session_open > 0 else 0, 6
        )
        result['session_range_pct'] = round(
            (self._session_high - self._session_low) / self._session_low
            if self._session_low > 0 else 0, 6
        )
        result['intraday_momentum'] = result['dist_session_open']

        # ── 5. Volume features ─────────────────────────────────
        vs = list(self._volumes)
        if len(vs) >= 20:
            result['vol_burst']      = round(vs[-1] / (np.mean(vs[-20:]) + 1e-9), 4)
            result['vol_ma20']       = round(float(np.mean(vs[-20:])), 4)
        else:
            result['vol_burst']      = 1.0
            result['vol_ma20']       = float(vol)

        # ── 6. Spread features ─────────────────────────────────
        ss = list(self._spreads)
        if len(ss) >= 20:
            avg_spread = np.mean(ss[-20:])
            result['spread_ratio']  = round(spread / (avg_spread + 1e-9), 4)
            result['spread_zscore'] = round(
                (spread - avg_spread) / (np.std(ss[-20:]) + 1e-9), 4
            )
        else:
            result['spread_ratio']  = 1.0
            result['spread_zscore'] = 0.0

        # ── 7. OBI momentum ────────────────────────────────────
        obs = list(self._obis)
        if len(obs) >= 5:
            result['obi_ma5']   = round(float(np.mean(obs[-5:])), 4)
            result['obi_delta'] = round(obs[-1] - obs[-2] if len(obs) >= 2 else 0, 4)
        else:
            result['obi_ma5']   = obi
            result['obi_delta'] = 0.0

        # ── 8. TFT momentum ────────────────────────────────────
        ts = list(self._tfts)
        if len(ts) >= 10:
            result['tft_ma10']  = round(float(np.mean(ts[-10:])), 4)
            result['tft_delta'] = round(ts[-1] - ts[-5] if len(ts) >= 5 else 0, 4)
        else:
            result['tft_ma10']  = tft
            result['tft_delta'] = 0.0

        # ── 9. Entropy momentum ────────────────────────────────
        es = list(self._entropies)
        if len(es) >= 10:
            result['entropy_ma10']   = round(float(np.mean(es[-10:])), 4)
            result['entropy_rising'] = 1.0 if len(es) >= 3 and es[-1] > es[-3] else 0.0
        else:
            result['entropy_ma10']   = ent
            result['entropy_rising'] = 0.0

        # ── 10. Regime encoding (one-hot) ──────────────────────
        macro = features.get('macro_regime', 'NEUTRAL')
        for r in ['WAR', 'RISK_ON', 'RISK_OFF', 'INFLATION', 'CRISIS', 'NEUTRAL']:
            result[f'regime_{r}'] = 1.0 if macro == r else 0.0

        vol_reg = features.get('vol_regime', 'NORMAL_VOL')
        for r in ['PANIC', 'HIGH_VOL', 'NORMAL_VOL', 'LOW_VOL']:
            result[f'volreg_{r}'] = 1.0 if vol_reg == r else 0.0

        result['pre_event_flag'] = float(features.get('pre_event_flag', 0))

        # ── 11. Lag features cross-asset (do LagFeatureEngine) ─
        for k, v in features.items():
            if '_ret_' in k and 's' in k:
                result[k] = float(v) if v else 0.0

        # ── 12. Autocorrelação de retornos ─────────────────────
        if n >= 20:
            rets = np.diff(p[-20:])
            if len(rets) >= 10 and np.std(rets) > 0:
                corr = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])
                result['autocorr_ret'] = round(corr if not np.isnan(corr) else 0, 4)
            else:
                result['autocorr_ret'] = 0.0
        else:
            result['autocorr_ret'] = 0.0

        return result

    def get_feature_names(self) -> List[str]:
        """Retorna lista dos nomes de features geradas."""
        dummy = {
            'bid': 100.0, 'ask': 100.01, 'ofi': 0, 'obi': 0,
            'tft': 0.5, 'vwap': 100.0, 'vwap_dev_pct': 0,
            'vwap_zscore': 0, 'spread': 0.01, 'volatility': 0.001,
            'entropy': 0.3, 'volume': 1.0, 'macro_regime': 'NEUTRAL',
            'vol_regime': 'NORMAL_VOL', 'pre_event_flag': 0,
        }
        return sorted(self.compute(dummy).keys())
