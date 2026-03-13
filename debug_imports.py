from brain.orchestrator import PITSOrchestrator
try:
    orch = PITSOrchestrator(dry_run=True)
    orch.initialize_engines()
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
