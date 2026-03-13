import logging
import os
from ml_engine.dataset_builder import DatasetBuilder
from ml_engine.bayesian_model import BayesianModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger("TrainingPipeline")
    
    print("="*60)
    print("      PITS - Predictive Intelligence Trading System      ")
    print("              ML ENGINE TRAINING PIPELINE                ")
    print("="*60)

    symbols = ["WTI", "XAUUSD", "US500", "DXY", "VIX", "BRENT"]
    horizon = 20 # Predict 20 ticks ahead
    
    # 1. Build Dataset
    builder = DatasetBuilder(data_path="data/ticks/")
    logger.info("Building dataset from historical Parquet files...")
    X, y = builder.get_all_data(symbols, horizon=horizon)
    
    if X.size == 0:
        logger.error("No data available for training. Ensure 'data/ticks/' contains Parquet files.")
        return

    # 2. Train Models
    # --- Bayesian ---
    logger.info("Training Bayesian Model...")
    bayes_model = BayesianModel()
    bayes_model.train(X, y)
    if bayes_model.is_trained:
        bayes_model.save_model("models/bayesian_model.pkl")

    # --- XGBoost ---
    logger.info("Training XGBoost Model...")
    xgb_model = XGBoostModel()
    xgb_model.train(X, y)
    if xgb_model.is_trained:
        xgb_model.save_model("models/xgboost_model.json")
    
    # 3. Persistence Check
    if bayes_model.is_trained and xgb_model.is_trained:
        print(f"\n[SUCCESS] Ensemble Training complete.")
        print(f"[INFO] Samples: {len(X)}")
        print(f"[INFO] Dataset balance: {y.mean()*100:.2f}% UP")
    else:
        logger.error("Training failed for one or more models.")

if __name__ == "__main__":
    main()
