import logging
from brain.orchestrator import PITSOrchestrator

def main():
    """
    PITS System Test (Dry Run)
    Runs the full pipeline without executing live trades.
    """
    print("="*60)
    print("      PITS - Predictive Intelligence Trading System      ")
    print("                PHASE 1 - DRY RUN TEST                  ")
    print("="*60)

    # Initialize Orchestrator in Dry Run mode
    orchestrator = PITSOrchestrator(dry_run=True)
    
    try:
        orchestrator.initialize_engines()
        print("\n[SYSTEM] Engines initialized. Watching market...")
        print("[SYSTEM] Signals will be logged below. Press Ctrl+C to stop.\n")
        
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupt received. Shutting down gracefully...")
        orchestrator.stop()
    except Exception as e:
        logging.error(f"System Crash: {str(e)}")
        orchestrator.stop()

if __name__ == "__main__":
    main()
