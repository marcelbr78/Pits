"""
run_pits_final.py — Entry point PITS completo (Fases 1-5)

Uso:
  python run_pits_final.py            # paper trading (seguro)
  python run_pits_final.py --live     # live trading real
  python run_pits_final.py --skip-lstm --skip-gnn  # sem modelos deep
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.orchestrator_final import PITSOrchestratorFinal

if __name__ == "__main__":
    live      = "--live"      in sys.argv
    skip_lstm = "--skip-lstm" in sys.argv
    skip_gnn  = "--skip-gnn"  in sys.argv

    print("=" * 60)
    print("  PITS — Predictive Intelligence Trading System")
    print("  Fases 1-5 | WTI/Oil | Exness MT5")
    print(f"  Modo: {'LIVE TRADING ⚠' if live else 'PAPER TRADING ✓'}")
    print("  Features: OBI | TFT | Lag | ATR | Pattern | LSTM | GNN")
    print("=" * 60)

    if live:
        confirm = input("\n  ⚠  LIVE TRADING ativado. Capital real em risco.\n  Digite CONFIRMAR para continuar: ")
        if confirm.strip() != "CONFIRMAR":
            print("  Cancelado.")
            sys.exit(0)

    orc = PITSOrchestratorFinal(dry_run=not live)
    orc.initialize_engines()
    orc.run()
