"""
Anomaly Detection — Fase 4.

Usa Isolation Forest para detectar:
  - Manipulação de mercado (spoofing)
  - Eventos extremos anômalos
  - Liquidez artificial
  - Flash crashes

Quando anomalia detectada:
  - Sistema pausa automaticamente
  - Log de alerta no dashboard
  - Aguarda 60 segundos antes de retomar
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List


class AnomalyDetector:
    """
    Isolation Forest simplificado para detecção de anomalias em tempo real.
    Usa uma implementação leve sem sklearn para não bloquear o loop de ticks.
    
    A versão completa com sklearn.IsolationForest é treinada offline
    e carregada como modelo salvo.
    """

    def __init__(self, contamination: float = 0.02, window: int = 200):
        self.logger = logging.getLogger("AnomalyDetector")
        self.contamination = contamination
        self.window = window

        self._history: deque = deque(maxlen=window)
        self._anomaly_count = 0
        self._last_anomaly_ts: float = 0.0
        self._pause_until: float = 0.0

        # Thresholds adaptativos (calculados sobre histórico)
        self._spread_threshold: Optional[float] = None
        self._vol_threshold: Optional[float] = None
        self._obi_threshold: Optional[float] = None

        # Tenta carregar modelo sklearn se disponível
        self._sklearn_model = self._try_load_model()

    def update(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa features atuais e detecta anomalias.
        
        Returns dict:
          is_anomaly: bool
          score: float (0-1, mais alto = mais anômalo)
          reason: str
          should_pause: bool
        """
        import time
        now = time.time()

        vec = self._extract_vector(features)
        self._history.append(vec)

        # Atualiza thresholds adaptativos
        if len(self._history) >= 50:
            self._update_thresholds()

        # Pausa ativa
        if now < self._pause_until:
            return {
                'is_anomaly': True,
                'score': 1.0,
                'reason': 'Pausa pós-anomalia ativa',
                'should_pause': True,
            }

        # Detecta anomalia
        result = self._detect(features, vec, now)

        if result['is_anomaly']:
            self._anomaly_count += 1
            self._last_anomaly_ts = now
            self._pause_until = now + 60  # pausa 60 segundos
            self.logger.warning(
                f"ANOMALIA detectada: {result['reason']} | "
                f"Score: {result['score']:.3f}"
            )

        return result

    def is_paused(self) -> bool:
        import time
        return time.time() < self._pause_until

    def _detect(self, features: Dict, vec: List[float], now: float) -> Dict:
        # Usa modelo sklearn se disponível
        if self._sklearn_model is not None:
            return self._detect_sklearn(vec)

        # Fallback: detecção por Z-score
        return self._detect_zscore(features)

    def _detect_zscore(self, features: Dict) -> Dict:
        """Detecção simples por Z-score nos valores atuais."""
        reasons = []
        max_score = 0.0

        spread  = features.get('spread', 0)
        volat   = features.get('volatility', 0)
        obi     = abs(features.get('obi', 0))
        entropy = features.get('entropy', 0)

        if self._spread_threshold and spread > self._spread_threshold * 3:
            score = min(1.0, spread / self._spread_threshold / 3)
            reasons.append(f"Spread anômalo: {spread:.5f} ({score:.1%})")
            max_score = max(max_score, score)

        if self._vol_threshold and volat > self._vol_threshold * 4:
            score = min(1.0, volat / self._vol_threshold / 4)
            reasons.append(f"Volatilidade extrema: {volat:.6f}")
            max_score = max(max_score, score)

        if obi > 0.95:
            score = (obi - 0.95) / 0.05
            reasons.append(f"OBI extremo: {obi:.3f}")
            max_score = max(max_score, min(1.0, score))

        if entropy < 0.05:
            score = 1.0 - entropy / 0.05
            reasons.append(f"Entropy zero — book sintético: {entropy:.4f}")
            max_score = max(max_score, score)

        threshold = 1 - self.contamination
        is_anomaly = max_score >= threshold

        return {
            'is_anomaly': is_anomaly,
            'score': round(max_score, 4),
            'reason': ' | '.join(reasons) if reasons else 'Normal',
            'should_pause': is_anomaly,
        }

    def _detect_sklearn(self, vec: List[float]) -> Dict:
        try:
            import numpy as np
            X = np.array(vec).reshape(1, -1)
            score = float(self._sklearn_model.decision_function(X)[0])
            pred  = int(self._sklearn_model.predict(X)[0])
            is_anomaly = pred == -1
            return {
                'is_anomaly': is_anomaly,
                'score': round(max(0, -score), 4),
                'reason': 'Isolation Forest detectou padrão incomum' if is_anomaly else 'Normal',
                'should_pause': is_anomaly,
            }
        except Exception as e:
            self.logger.warning(f"Erro no modelo sklearn: {e}")
            return {'is_anomaly': False, 'score': 0.0, 'reason': 'Erro modelo', 'should_pause': False}

    def _update_thresholds(self):
        hist = list(self._history)
        spreads = [v[0] for v in hist]
        vols    = [v[1] for v in hist]
        obis    = [abs(v[2]) for v in hist]

        self._spread_threshold = float(np.mean(spreads)) + 2 * float(np.std(spreads))
        self._vol_threshold    = float(np.mean(vols))    + 2 * float(np.std(vols))
        self._obi_threshold    = float(np.mean(obis))    + 2 * float(np.std(obis))

    def _extract_vector(self, features: Dict) -> List[float]:
        return [
            features.get('spread', 0),
            features.get('volatility', 0),
            features.get('obi', 0),
            features.get('entropy', 0),
            features.get('tft', 0.5),
            abs(features.get('vwap_dev_pct', 0)),
        ]

    def _try_load_model(self):
        try:
            import joblib, os
            path = 'models/anomaly_detector.pkl'
            if os.path.exists(path):
                model = joblib.load(path)
                self.logger.info("Modelo Isolation Forest carregado.")
                return model
        except Exception:
            pass
        return None

    def train_and_save(self, feature_vectors: List[List[float]]):
        """Treina Isolation Forest offline e salva."""
        try:
            from sklearn.ensemble import IsolationForest
            import joblib, os
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42, n_estimators=100
            )
            X = np.array(feature_vectors)
            model.fit(X)
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, 'models/anomaly_detector.pkl')
            self._sklearn_model = model
            self.logger.info(f"Isolation Forest treinado com {len(X)} amostras e salvo.")
        except ImportError:
            self.logger.warning("sklearn não disponível — usando detecção por Z-score.")
        except Exception as e:
            self.logger.error(f"Erro ao treinar anomaly detector: {e}")
