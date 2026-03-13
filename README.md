# PITS — Predictive Intelligence Trading System

PITS is a modular quantitative trading platform designed to harness market microstructure, machine learning, and multi-asset correlation for predictive trading.

## Core Assets
- **WTI** & **Brent** (Crude Oil)
- **Gold** (XAUUSD)
- **SP500** (Equity Index)
- **DXY** (US Dollar Index)
- **VIX** (Volatility Index)

## Architecture
The system follows a central **Brain Orchestrator** pattern that coordinates independent modular engines:
- `brain/`: Orchestration and lifecycle management.
- `data_engine/`: Multi-source data ingestion (MT5, Bookmap, etc.).
- `feature_engine/`: Statistical and microstructural feature calculation.
- `ml_engine/`: Predictive modeling (Bayesian, XGBoost, GNN).
- `risk_engine/`: Portfolio management and exposure control.
- `execution_engine/`: Smart order execution (TWAP, Liquidity Seeking).
- `learning_engine/`: Continuous model optimization.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/pits.git
cd pits

# Install dependencies
pip install -r requirements.txt
```

## Documentation
See the `docs/` folder for detailed technical blueprints and development guides.
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [AGENTS.md](AGENTS.md)
- [ROADMAP.md](ROADMAP.md)
