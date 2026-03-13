import logging
from typing import Dict, Any, List
from .ofi_calculator import OFICalculator
from .obi_calculator import OBICalculator
from .trade_flow import TradeFlowAnalyzer
from .vwap_calculator import VWAPCalculator
from .vwap_deviation import VWAPDeviationCalculator
from .volatility_calculator import VolatilityCalculator
from .entropy_calculator import EntropyCalculator
from .lag_features import LagFeatureEngine

class FeaturePipelineV2:
    """
    Feature Pipeline Fase 2 — versão expandida.
    
    Novo vetor de features por tick:
      symbol, timestamp
      ofi          : OFI original (fallback quando DOM indisponível)
      obi          : Order Book Imbalance 10 níveis (novo)
      tft          : Trade Flow Toxicity — aggressor side (novo)
      vwap         : VWAP da sessão
      vwap_dev_pct : Desvio percentual do VWAP (novo)
      vwap_zscore  : Z-score do desvio (novo)
      spread       : spread absoluto
      volatility   : realized volatility
      entropy      : Shannon entropy
      lag_*        : cross-asset lag features (novo — 15 features)
    
    Total: ~25 features vs 7 da Fase 1.
    """

    def __init__(self, symbols: List[str]):
        self.logger = logging.getLogger("FeaturePipelineV2")
        self.symbols = symbols
        self.calculators: Dict[str, Dict[str, Any]] = {}
        self.lag_engine = LagFeatureEngine(max_history_seconds=120)

        for symbol in symbols:
            self.calculators[symbol] = {
                'ofi':      OFICalculator(),
                'obi':      OBICalculator(levels=10),
                'tft':      TradeFlowAnalyzer(window_seconds=60),
                'vwap':     VWAPCalculator(),
                'vwap_dev': VWAPDeviationCalculator(),
                'vol':      VolatilityCalculator(window_size=100),
                'entropy':  EntropyCalculator(window_size=50),
            }

    def process_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa tick e retorna vetor completo de features Fase 2.
        Compatível com o formato da Fase 1 — adiciona campos sem remover.
        """
        symbol = tick['symbol']

        # Atualiza lag engine com este ativo
        self.lag_engine.update(symbol, tick)

        if symbol not in self.calculators:
            return tick

        c = self.calculators[symbol]

        # Features básicas (Fase 1 — mantidas)
        ofi    = round(c['ofi'].update(tick), 4)
        vwap   = round(c['vwap'].update(tick), 4)
        spread = round(tick.get('spread', tick['ask'] - tick['bid']), 6)
        vol    = round(c['vol'].update(tick), 8)
        ent    = round(c['entropy'].update(tick), 6)

        # Features novas (Fase 2)
        obi       = c['obi'].calculate(symbol)
        tft       = c['tft'].update(tick)
        vwap_info = c['vwap_dev'].update(tick)

        # Lag features cross-asset
        ts_ms = tick.get('time_msc', tick.get('timestamp', 0) * 1000)
        lag_feats = self.lag_engine.get_lag_features(current_ts_ms=ts_ms)

        features = {
            # Identificação
            'symbol':    symbol,
            'timestamp': tick.get('timestamp'),
            'bid':       tick.get('bid'),
            'ask':       tick.get('ask'),

            # Fase 1 — mantidas
            'ofi':        ofi,
            'vwap':       vwap,
            'spread':     spread,
            'volatility': vol,
            'entropy':    ent,

            # Fase 2 — novas
            'obi':          obi,
            'tft':          tft,
            'tft_signal':   c['tft'].get_signal(tft),
            'vwap_dev_pct': vwap_info['deviation_pct'],
            'vwap_zscore':  vwap_info['z_score'],
            'dom_available': c['obi'].is_dom_available(),

            # Lag features (15-20 features adicionais)
            **lag_feats,
        }

        return features

    def reset_sessions(self):
        """Reset diário de sessão."""
        for symbol in self.symbols:
            self.calculators[symbol]['vwap'].reset()
        self.logger.info("Sessions reset.")
