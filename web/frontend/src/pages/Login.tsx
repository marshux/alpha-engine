export function Login() {
  const params = new URLSearchParams(window.location.search);
  const unauthorized = params.get("error") === "unauthorized";

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-6 py-12 text-slate-100">
      <div className="flex w-full max-w-3xl flex-col gap-8 md:flex-row md:items-start">
        <div className="flex-1">
          <h1 className="mb-3 text-2xl font-semibold">alpha-engine</h1>
          <p className="mb-4 text-sm leading-relaxed text-slate-400">
            A systematic swing-trading research engine: quant signals (momentum, mean-reversion,
            volatility, value, quality) feed a deterministic decision engine, backtested against
            SPY/VOO/VTI over days-to-weeks holding periods. Backtest-first -- no live/paper order
            execution yet.
          </p>
          <p className="mb-4 text-sm leading-relaxed text-slate-400">
            <span className="font-medium text-slate-300">AI is allowed to research and score; it is
            never allowed to decide.</span> Everything downstream of signal generation -- ranking,
            sizing, risk limits -- is a plain rules pipeline with no model call anywhere in it, so a
            backtest given the same inputs always produces the same trades.
          </p>
          <p className="text-sm leading-relaxed text-slate-400">
            This dashboard tracks backtest and walk-forward run history: equity curves, return
            metrics, and tear sheets. A{" "}
            <a href="/public" className="text-emerald-400 hover:underline">
              public view
            </a>{" "}
            shows aggregate returns only -- no positions, trades, or strategy internals -- delayed
            by 30 days.
          </p>
        </div>

        <div className="w-full max-w-sm shrink-0 rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center">
          <h2 className="mb-1 text-lg font-semibold">Owner sign-in</h2>
          <p className="mb-6 text-sm text-slate-400">Private -- single-user.</p>
          {unauthorized && (
            <p className="mb-4 rounded-md border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-300">
              That GitHub account isn't authorized for this dashboard.
            </p>
          )}
          <a
            href="/auth/login"
            className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-slate-100 px-4 py-2 font-medium text-slate-900 hover:bg-white"
          >
            Sign in with GitHub
          </a>
          <a href="/public" className="mt-4 block text-sm text-slate-400 hover:text-white">
            View public dashboard →
          </a>
        </div>
      </div>
    </div>
  );
}
