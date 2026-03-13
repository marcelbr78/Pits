from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Dict, Any
from .state import SystemState

app = FastAPI(title="PITS Real-Time API")
state: SystemState = None # Will be injected by orchestrator

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    s = state.get_full_state()
    return {
        "is_running": s["is_running"],
        "is_live": s["is_live"],
        "mt5_connected": s["mt5_connected"]
    }

@app.get("/metrics")
async def get_metrics():
    return state.get_full_state()["metrics"]

@app.get("/signals")
async def get_signals():
    return state.get_full_state()["signals"]

@app.get("/positions")
async def get_positions():
    return state.get_full_state()["positions"]

@app.get("/trades")
async def get_trades():
    return state.get_full_state()["last_trades"]

@app.get("/regime")
async def get_regime():
    return state.get_full_state()["regime"]

@app.get("/features/{symbol}")
async def get_features(symbol: str):
    features = state.get_full_state()["features"]
    return features.get(symbol, {})

@app.get("/calendar")
async def get_calendar():
    return state.get_full_state()["next_event"]

@app.post("/control")
async def control_system(payload: Dict[str, str] = Body(...)):
    action = payload.get("action")
    # This will be handled by the orchestrator watching the state
    if action == "pause":
        state.set_running(False)
    elif action == "resume":
        state.set_running(True)
    elif action == "set_live":
        state.set_live(True)
    elif action == "set_paper":
        state.set_live(False)
    
    return {"status": "success", "action": action}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Broadcast state every 1 second
            full_state = state.get_full_state()
            await websocket.send_text(json.dumps(full_state))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

def run_server(shared_state: SystemState, host: str = "0.0.0.0", port: int = 8001):
    import uvicorn
    global state
    state = shared_state
    uvicorn.run(app, host=host, port=port, log_level="error")
