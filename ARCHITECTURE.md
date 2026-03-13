# PITS Technical Architecture

## Overview
PITS (Predictive Intelligence Trading System) is designed using a **Clean Architecture** approach, separating concerns into specialized engines coordinated by a central orchestrator.

## Components

### 1. Brain (Orchestrator)
The central nervous system. It initializes all components, manages the global state, and ensures data flow between engines.

### 2. Data Engine
Responsible for real-time and historical data gathering.
- **MT5 API**: Core execution and price feed.
- **Bookmap**: Depth of Market (DOM) / Order Book data.
- **Alternative Data**: News API, Economic Calendars.

### 3. Feature Engine
Transforms raw data into predictive signals.
- **Phase 1**: OFI (Order Flow Imbalance), Shannon Entropy, ATR.
- **Phase 2**: OBI (Order Book Imbalance), Aggressor Side (Toxicity).
- **Phase 3**: Markov Regimes, Cross-asset Lead-lag.

### 4. ML Engine
Layers of intelligence for price prediction.
- **Base Layer**: Naive Bayes / Bayesian Inference.
- **Advanced Layer**: XGBoost / Random Forest (Feature Importance).
- **Temporal Layer**: LSTM / Transformer (Time-series attention).
- **Relational Layer**: GNN (Graph Neural Networks) for asset correlations.

### 5. Risk Engine
Protects capital and manages exposure.
- **Mean-Variance Optimization**: Markowitz portfolio theory.
- **Monte Carlo**: 10,000 simulations for ruin probability.
- **Kelly Criterion**: Dynamic position sizing.

### 6. Execution Engine
Minimizes slippage and market impact.
- **TWAP**: Time Weighted Average Price for large orders.
- **Liquidity Seeking**: Executing only when market conditions are optimal.

## Data Flow
`Market Data` -> `Data Engine` -> `Feature Engine` -> `ML Engine` -> `Brain` -> `Risk Engine` -> `Execution Engine` -> `Market`
