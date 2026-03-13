"""
API Server V2 — endpoints adicionais para Fases 3-5.

Adiciona ao servidor existente:
  GET /calendar     → próximo evento econômico
  GET /logs         → logs do sistema
  GET /features/{symbol} → features detalhadas Fase 2+
  GET /anomaly      → status do anomaly detector
  GET /portfolio    → resumo do portfolio risk engine

Este arquivo EXTENDE o server.py existente.
Para usar, faça create_app() importar add_extra_routes().
"""

from fastapi import FastAPI
from typing import Dict, Any


def add_extra_routes(app: FastAPI, shared_state):
    """Adiciona rotas extras ao app FastAPI existente."""

    @app.get("/calendar")
    def get_calendar():
        """Retorna próximo evento econômico."""
        with shared_state.lock:
            return shared_state.next_event

    @app.get("/logs")
    def get_logs():
        """Retorna logs do sistema."""
        with shared_state.lock:
            return {"logs": shared_state.logs}

    @app.get("/features/{symbol}")
    def get_features(symbol: str):
        """Retorna features detalhadas de um ativo."""
        with shared_state.lock:
            return {
                "symbol": symbol,
                "features": shared_state.features.get(symbol, {})
            }

    @app.get("/portfolio")
    def get_portfolio():
        """Retorna resumo do portfolio risk."""
        with shared_state.lock:
            return {
                "positions": shared_state.positions,
                "metrics": shared_state.metrics,
            }

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0-phases1-5"}
