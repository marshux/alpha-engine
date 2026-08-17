"""Full-universe insider_buying backfill, continuing from wherever
data/cache/insider/*.parquet already stands (already-cached symbols are
skipped automatically by get_insider_transactions's own cache check)."""
import time

from alpha_engine.config import load_universe, universe_symbols
from alpha_engine.data.insider import get_insider_transactions

universe = load_universe()
symbols = universe_symbols(universe)
print(f"Backfilling insider transactions for {len(symbols)} symbols...", flush=True)

t0 = time.time()
done, failed, total_txns = 0, 0, 0
for i, symbol in enumerate(symbols):
    sym_t0 = time.time()
    try:
        df = get_insider_transactions(symbol)
        total_txns += len(df)
        done += 1
        print(
            f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} txns "
            f"({time.time()-sym_t0:.1f}s, {time.time()-t0:.0f}s elapsed total)",
            flush=True,
        )
    except Exception as e:
        failed += 1
        print(f"[{i+1}/{len(symbols)}] {symbol}: FAILED -- {e}", flush=True)

elapsed = time.time() - t0
print(
    f"\nDONE in {elapsed:.0f}s -- {done} symbols cached, {failed} failed, "
    f"{total_txns} total transactions",
    flush=True,
)
