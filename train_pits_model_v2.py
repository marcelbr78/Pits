"""
Training Pipeline Completo — Fases 4 e 5.

Treina todos os modelos em sequência:
  1. Dataset Builder V2 — 100+ features
  2. XGBoost — gradiente boosting
  3. Bayesian — GaussianNB
  4. LSTM — sequências temporais
  5. GNN — multi-ativo (se PyG disponível)
  6. Anomaly Detector — Isolation Forest
  7. Salva todos os modelos
  8. Roda Monte Carlo para validar segurança

Uso:
  python train_pits_model_v2.py
  python train_pits_model_v2.py --symbol USOILm
  python train_pits_model_v2.py --skip-lstm  (se sem GPU)
"""

import sys
import os
import logging
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingPipelineV2")


def train_all(
    symbols=None,
    skip_lstm=False,
    skip_gnn=False,
    horizon=20,
):
    if symbols is None:
        symbols = ["USOILm", "XAUUSDm", "UKOILm"]

    logger.info("=" * 60)
    logger.info("  PITS Training Pipeline V2 — Fases 4 e 5")
    logger.info("=" * 60)

    # ── 1. Dataset ────────────────────────────────────────────
    logger.info("\n[1/6] Construindo dataset com 100+ features...")
    from ml_engine.dataset_builder_v2 import DatasetBuilderV2
    builder = DatasetBuilderV2()
    X, y, feat_names = builder.build_all(symbols, horizon=horizon)

    if X.size == 0:
        logger.error("Dataset vazio — verifique se há dados em data/ticks/")
        return False

    logger.info(f"Dataset: {len(X)} amostras × {len(feat_names)} features")
    logger.info(f"Balance: UP={y.mean()*100:.1f}% | DOWN={(1-y.mean())*100:.1f}%")

    # Split treino/validação
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    builder.save_scaler()

    # ── 2. XGBoost ────────────────────────────────────────────
    logger.info("\n[2/6] Treinando XGBoost...")
    try:
        from ml_engine.xgboost_model import XGBoostModel
        xgb = XGBoostModel()
        xgb.train(X_train, y_train)
        xgb.save_model()

        val_acc = xgb.model.score(X_val, y_val)
        logger.info(f"XGBoost validação: {val_acc:.4f} ({val_acc*100:.1f}%)")

        # Feature importance
        try:
            importance = xgb.model.feature_importances_
            top10_idx = np.argsort(importance)[-10:][::-1]
            logger.info("Top 10 features mais importantes:")
            for idx in top10_idx:
                if idx < len(feat_names):
                    logger.info(f"  {feat_names[idx]}: {importance[idx]:.4f}")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"XGBoost falhou: {e}")

    # ── 3. Bayesian ───────────────────────────────────────────
    logger.info("\n[3/6] Treinando Bayesian (GaussianNB)...")
    try:
        from ml_engine.bayesian_model import BayesianModel
        bayes = BayesianModel()
        # Usa primeiras 5 features (compatibilidade Fase 1)
        X_bayes = X_train[:, :5] if X_train.shape[1] >= 5 else X_train
        bayes.model.fit(X_bayes, y_train)
        bayes.is_trained = True
        bayes.save_model()

        X_val_bayes = X_val[:, :5] if X_val.shape[1] >= 5 else X_val
        val_acc = bayes.model.score(X_val_bayes, y_val)
        logger.info(f"Bayesian validação: {val_acc:.4f} ({val_acc*100:.1f}%)")

    except Exception as e:
        logger.error(f"Bayesian falhou: {e}")

    # ── 4. LSTM ───────────────────────────────────────────────
    if not skip_lstm:
        logger.info("\n[4/6] Treinando LSTM (sequências temporais)...")
        try:
            from ml_engine.lstm_model import LSTMModel
            seq_len = 60
            n_feat  = min(25, X.shape[1])

            # Cria sequências
            seqs, seq_labels = [], []
            for i in range(seq_len, len(X)):
                seqs.append(X[i-seq_len:i, :n_feat])
                seq_labels.append(y[i])

            if seqs:
                seqs_arr   = np.array(seqs, dtype=np.float32)
                labels_arr = np.array(seq_labels, dtype=np.float32)

                lstm = LSTMModel(n_features=n_feat, seq_length=seq_len)
                lstm.train(seqs_arr[:int(len(seqs_arr)*0.8)],
                           labels_arr[:int(len(labels_arr)*0.8)],
                           epochs=30)
                logger.info("LSTM treinado e salvo.")
            else:
                logger.warning("Dados insuficientes para LSTM.")

        except Exception as e:
            logger.error(f"LSTM falhou: {e}")
    else:
        logger.info("\n[4/6] LSTM — pulado (--skip-lstm)")

    # ── 5. GNN ────────────────────────────────────────────────
    if not skip_gnn:
        logger.info("\n[5/6] Treinando GNN (multi-ativo)...")
        try:
            from ml_engine.gnn_model import TradingGNN
            gnn = TradingGNN(n_features_per_node=8)
            if gnn._pyg_available:
                n_feat_gnn = min(8, X.shape[1])
                gnn.train(
                    X_train[:, :n_feat_gnn].reshape(-1, 1, n_feat_gnn),
                    y_train, epochs=30
                )
                logger.info("GNN treinada e salva.")
            else:
                logger.warning("PyTorch Geometric não disponível — GNN usa fallback.")
        except Exception as e:
            logger.error(f"GNN falhou: {e}")
    else:
        logger.info("\n[5/6] GNN — pulado (--skip-gnn)")

    # ── 6. Anomaly Detector ───────────────────────────────────
    logger.info("\n[6/6] Treinando Anomaly Detector (Isolation Forest)...")
    try:
        from market_intelligence.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector()
        feat_vecs = X_train[:, :6].tolist()
        detector.train_and_save(feat_vecs)
        logger.info("Anomaly Detector treinado e salvo.")
    except Exception as e:
        logger.error(f"Anomaly Detector falhou: {e}")

    # ── 7. Monte Carlo ────────────────────────────────────────
    logger.info("\n[Bônus] Rodando Monte Carlo 10.000 simulações...")
    try:
        from risk_engine.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=10000)

        # Estima win rate e PnL médio pelo validação
        try:
            from ml_engine.xgboost_model import XGBoostModel as XGB2
            xgb2 = XGB2()
            xgb2.load_model()
            preds = xgb2.model.predict(X_val)
            wins  = (preds == y_val).sum()
            win_rate  = wins / len(y_val)
        except Exception:
            win_rate = 0.55

        result = mc.quick_check(
            win_rate=win_rate,
            avg_win=0.38,
            avg_loss=0.24,
            capital=30.0,
            n_trades=100,
        )
        logger.info(
            f"Monte Carlo: P(ruína)={result['p_ruin_pct']}% | "
            f"Capital mediana=${result['capital_median']} | "
            f"Drawdown p95={result['drawdown_p95']:.1%} | "
            f"{'✓ SEGURO' if result['safe'] else '⚠ RISCO ALTO'}"
        )
    except Exception as e:
        logger.error(f"Monte Carlo falhou: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("  Treino completo. Modelos salvos em models/")
    logger.info("  Inicie com: python run_system_test_v2.py")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PITS Training Pipeline V2")
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--skip-lstm', action='store_true')
    parser.add_argument('--skip-gnn',  action='store_true')
    parser.add_argument('--horizon',   type=int, default=20)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    train_all(
        symbols=symbols,
        skip_lstm=args.skip_lstm,
        skip_gnn=args.skip_gnn,
        horizon=args.horizon,
    )
