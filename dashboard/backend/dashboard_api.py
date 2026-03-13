from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import json
from paper_trading.performance_tracker import PerformanceTracker

app = FastAPI(title="PITS Monitoring Dashboard")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
TRADES_PATH = "data/paper_trades.parquet"
TICKS_DIR = "data/ticks/"

@app.get("/api/trades")
async def get_trades():
    if os.path.exists(TRADES_PATH):
        df = pd.read_parquet(TRADES_PATH)
        # Convert timestamps for JSON serialization
        if 'timestamp_entry' in df.columns:
            df['timestamp_entry'] = df['timestamp_entry'].astype(str)
        if 'timestamp_exit' in df.columns:
            df['timestamp_exit'] = df['timestamp_exit'].astype(str)
        return df.to_dict(orient="records")
    return []

@app.get("/api/performance")
async def get_performance():
    if os.path.exists(TRADES_PATH):
        df = pd.read_parquet(TRADES_PATH)
        tracker = PerformanceTracker()
        return tracker.calculate_metrics(df)
    return {}

@app.get("/api/market_state")
async def get_market_state():
    # This would ideally read from a shared state or latest logs
    # For now, we mock it based on live log structure expectation 
    # or return a generic response since the orchestrator runs in a separate process.
    return {
        "status": "Operational",
        "symbols": ["WTI", "XAUUSD", "US500", "DXY", "VIX", "BRENT"]
    }

@app.get("/api/predictions")
async def get_predictions():
    # Placeholder for latest probabilities
    return {"latest": "Updating..."}

# Serve frontend
# Note: In a real deploy, we'd use StaticFiles
# app.mount("/", StaticFiles(directory="dashboard/frontend", html=True), name="frontend")
