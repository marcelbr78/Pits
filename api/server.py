from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json
from api.state import SystemState
from api.mcp_bridge import setup_mcp_bridge

def create_app(shared_state: SystemState) -> FastAPI:
    app = FastAPI(title="PITS API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    setup_mcp_bridge(app, shared_state)

    @app.get("/status")
    def get_status():
        with shared_state.lock:
            return {
                "is_running": shared_state.is_running,
                "is_live": shared_state.is_live,
                "mt5_connected": shared_state.mt5_connected,
                "ticks_per_second": shared_state.ticks_per_second
            }

    @app.get("/metrics")
    def get_metrics():
        with shared_state.lock:
            return shared_state.metrics

    @app.get("/signals")
    def get_signals():
        with shared_state.lock:
            return shared_state.signals

    @app.get("/positions")
    def get_positions():
        with shared_state.lock:
            return shared_state.positions

    @app.get("/trades")
    def get_trades():
        with shared_state.lock:
            return shared_state.last_trades

    @app.get("/regime")
    def get_regime():
        with shared_state.lock:
            return shared_state.regime

    @app.get("/features/{symbol}")
    def get_features(symbol: str):
        with shared_state.lock:
            return shared_state.features.get(symbol.upper(), {})

    @app.get("/calendar")
    def get_calendar():
        with shared_state.lock:
            return shared_state.next_event

    @app.post("/control")
    def control(action: str):
        if action == "pause":
            shared_state.set_running(False)
        elif action == "resume":
            shared_state.set_running(True)
        elif action == "set_live":
            with shared_state.lock:
                shared_state.is_live = True
        return {"status": "ok", "action": action}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = shared_state.get_full_state()
                await websocket.send_text(json.dumps(data, default=str))
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

    @app.get("/dashboard", response_class=HTMLResponse)
    def get_dashboard():
        import os
        dashboard_path = "dashboard/index.html"
        if os.path.exists(dashboard_path):
            with open(dashboard_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Dashboard file not found."

    return app

def run_server(shared_state: SystemState, host="0.0.0.0", port=8001):
    import uvicorn
    app = create_app(shared_state)
    uvicorn.run(app, host=host, port=port, log_level="error")
