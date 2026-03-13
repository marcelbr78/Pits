import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"Failed to initialize MT5: {mt5.last_error()}")
    quit()

symbols = mt5.symbols_get()
print(f"Total symbols found: {len(symbols)}")

# List top 50 symbols for analysis
for i, s in enumerate(symbols[:50]):
    print(f"{i}: {s.name}")

mt5.shutdown()
