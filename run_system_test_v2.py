"""
run_system_test_v2.py — Fase 2

Inicia o sistema com todos os componentes da Fase 2.
Modo dry_run=True por padrão — não executa trades reais.

Uso:
  python run_system_test_v2.py
  python run_system_test_v2.py --live   (ativa live trading)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.orchestrator_v2 import PITSOrchestratorV2

if __name__ == "__main__":
    live = "--live" in sys.argv
    
    print("=" * 60)
    print("  PITS V2 — Fase 2")
    print("  OBI 10 níveis | TFT | Lag Features | Calendário")
    print(f"  Modo: {'LIVE TRADING' if live else 'PAPER TRADING (dry run)'}")
    print("=" * 60)

    orchestrator = PITSOrchestratorV2(dry_run=not live)
    orchestrator.initialize_engines()
    orchestrator.run()
