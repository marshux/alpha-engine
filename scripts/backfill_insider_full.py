"""Full-universe insider_buying backfill, continuing from wherever
data/cache/insider/*.parquet already stands (already-cached symbols are
skipped automatically by get_insider_transactions's own cache check).

Uses a small thread pool (see MAX_WORKERS below) so multiple symbols'
network-bound fetches overlap instead of running strictly one at a time --
safe to do because rate limiting now lives centrally in
data/insider.py::_throttle(), a thread-safe gate shared by every
_get_json/_get_text call regardless of which thread issues it. This script
doesn't need to know concurrency is happening at all; it just calls
get_insider_transactions(symbol) per symbol same as before, from multiple
threads via ThreadPoolExecutor.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from alpha_engine.config import load_universe, universe_symbols
from alpha_engine.data.insider import get_insider_transactions

# _throttle() caps aggregate request-issue rate at ~8.5 req/sec regardless
# of thread count -- this just needs enough concurrent threads that the
# pipeline can actually approach that ceiling (a single thread's own
# non-network overhead, e.g. XML parsing, would otherwise leave the
# request pipe idle between calls rather than saturating the rate cap).
MAX_WORKERS = 3

universe = load_universe()
symbols = universe_symbols(universe)
print(f"Backfilling insider transactions for {len(symbols)} symbols...", flush=True)

t0 = time.time()
done, failed, total_txns = 0, 0, 0


def _fetch(symbol: str):
    sym_t0 = time.time()
    df = get_insider_transactions(symbol)
    return df, time.time() - sym_t0


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_symbol = {executor.submit(_fetch, symbol): symbol for symbol in symbols}
    for i, future in enumerate(as_completed(future_to_symbol)):
        symbol = future_to_symbol[future]
        try:
            df, sym_elapsed = future.result()
            total_txns += len(df)
            done += 1
            print(
                f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} txns "
                f"({sym_elapsed:.1f}s, {time.time()-t0:.0f}s elapsed total)",
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
