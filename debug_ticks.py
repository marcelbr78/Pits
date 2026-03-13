import MetaTrader5 as mt5
import datetime
import pandas as pd

if not mt5.initialize():
    print("Initialize failed")
    quit()

symbol = "XAUUSDm"
mt5.symbol_select(symbol, True)

# Try fetching just 100 ticks from now
ticks = mt5.copy_ticks_from(symbol, datetime.datetime.now(), 100, mt5.COPY_TICKS_ALL)

if ticks is None:
    print(f"Failed to fetch: {mt5.last_error()}")
else:
    print(f"Fetched {len(ticks)} ticks for {symbol}")
    if len(ticks) > 0:
        df = pd.DataFrame(ticks)
        print(df.head())

mt5.shutdown()
