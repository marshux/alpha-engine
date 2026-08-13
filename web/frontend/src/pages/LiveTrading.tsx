import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../api";
import { buildEquityChartData } from "../chartData";
import { Pagination, usePagination } from "../components/Pagination";
import { currency, num, pct } from "../format";
import type { AiUsageSummary, BenchmarkPnl, LiveAccount, RunDetail } from "../types";

const POSITIONS_POLL_MS = 10_000;
const STRATEGY_COLOR = "#34d399";
const PAGE_SIZE = 10;

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}

export function LiveTrading() {
  const [session, setSession] = useState<RunDetail | null>(null);
  const [account, setAccount] = useState<LiveAccount | null>(null);
  const [aiUsage, setAiUsage] = useState<AiUsageSummary | null>(null);
  const [benchmarkPnl, setBenchmarkPnl] = useState<BenchmarkPnl | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionPending, setActionPending] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadPositions = () => {
    api
      .getLivePositions()
      .then(setAccount)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  };

  const loadSession = async () => {
    const summary = await api.startPaperTrading(); // idempotent get-or-create
    const detail = await api.getRun(summary.id);
    setSession(detail);
  };

  const loadAiUsage = () => {
    // Doesn't change fast (at most once/day) -- fetched on mount and after
    // actions that could trigger a fresh news_ai call, not on the 10s
    // positions poll.
    api
      .getAiUsage()
      .then(setAiUsage)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  };

  const loadBenchmarkPnl = () => {
    // Same cadence rationale as loadAiUsage -- the underlying equity
    // baseline only moves on a rebalance, and benchmark closes are daily,
    // not worth refetching on the 10s positions poll.
    api
      .getBenchmarkPnl()
      .then(setBenchmarkPnl)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  };

  useEffect(() => {
    loadSession().catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    loadPositions();
    loadAiUsage();
    loadBenchmarkPnl();
    timerRef.current = setInterval(loadPositions, POSITIONS_POLL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleRebalanceNow = async () => {
    setActionPending("rebalance");
    setError(null);
    try {
      await api.rebalanceNow();
      // The rebalance runs in the background (can take up to ~a couple
      // minutes -- real Alpaca + Anthropic calls) -- refresh once now for
      // immediate feedback if it was fast, the regular poll picks up the
      // rest.
      setTimeout(() => {
        loadPositions();
        loadSession();
        loadAiUsage();
        loadBenchmarkPnl();
      }, 3000);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setActionPending(null);
    }
  };

  const handleCheckAnomalies = async () => {
    setActionPending("anomaly-check");
    setError(null);
    try {
      await api.checkAnomaliesNow();
      setTimeout(() => {
        loadPositions();
        loadSession();
      }, 3000);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setActionPending(null);
    }
  };

  const handleFlatten = async () => {
    if (!window.confirm("Liquidate every position and cancel all open orders immediately?")) {
      return;
    }
    setActionPending("flatten");
    setError(null);
    try {
      await api.flattenToCash();
      setTimeout(loadPositions, 2000);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setActionPending(null);
    }
  };

  // Hooks must run unconditionally on every render -- computed here, above
  // the loading/error early returns below, with empty-array fallbacks
  // before session/account have loaded.
  const orders = session ? [...session.live_orders].reverse() : []; // newest first
  const positionsPage = usePagination(account?.positions ?? [], PAGE_SIZE);
  const ordersPage = usePagination(orders, PAGE_SIZE);

  if (error) return <p className="text-red-400">Error: {error}</p>;
  if (session === null || account === null) return <p className="text-slate-400">Loading...</p>;

  const equityData = session.equity_points.length > 0 ? buildEquityChartData(session) : [];

  // Baseline is the earliest recorded equity point (the account's equity
  // right after the first rebalance, before which no capital had been
  // deployed) -- not a hardcoded starting-capital assumption, since Alpaca
  // paper accounts can be seeded with any amount.
  const startingEquity = session.equity_points.length > 0 ? session.equity_points[0].equity : null;
  const pnlDollars = startingEquity !== null ? account.equity - startingEquity : null;
  const pnlPercent = startingEquity ? account.equity / startingEquity - 1 : null;
  const pnlPositive = pnlDollars !== null && pnlDollars >= 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Paper trading</h1>
          <p className="text-sm text-slate-400">
            Session started {session.start_date} -- Alpaca paper account
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRebalanceNow}
            disabled={actionPending !== null}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {actionPending === "rebalance" ? "Triggering..." : "Rebalance now"}
          </button>
          <button
            onClick={handleCheckAnomalies}
            disabled={actionPending !== null}
            className="rounded-md bg-slate-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-600 disabled:opacity-50"
          >
            {actionPending === "anomaly-check" ? "Checking..." : "Check for anomalies"}
          </button>
          <button
            onClick={handleFlatten}
            disabled={actionPending !== null}
            className="rounded-md bg-red-900 px-3 py-1.5 text-sm font-medium text-red-200 hover:bg-red-800 disabled:opacity-50"
          >
            {actionPending === "flatten" ? "Flattening..." : "Flatten to cash"}
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <div className="text-xs uppercase tracking-wide text-slate-500">
          Total P/L since {session.start_date}
        </div>
        {pnlDollars === null ? (
          <div className="mt-2 text-3xl font-bold text-slate-500">No rebalances yet</div>
        ) : (
          <div className="mt-2 flex flex-wrap items-baseline gap-3">
            <span className={`text-4xl font-bold ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}>
              {pnlPositive ? "+" : "-"}
              {currency(Math.abs(pnlDollars))}
            </span>
            <span className={`text-xl font-semibold ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}>
              ({pnlPositive ? "+" : "-"}
              {pct(Math.abs(pnlPercent ?? 0))})
            </span>
            <span className="text-sm text-slate-500">vs. {currency(startingEquity)} starting equity</span>
          </div>
        )}
        {benchmarkPnl && benchmarkPnl.benchmarks.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-4 border-t border-slate-800 pt-3">
            <span className="text-xs text-slate-500">If invested in an index fund instead:</span>
            {benchmarkPnl.benchmarks.map((b) => {
              const positive = b.pnl_usd >= 0;
              return (
                <span key={b.symbol} className="text-sm">
                  <span className="text-slate-400">{b.symbol}: </span>
                  <span className={positive ? "text-emerald-400" : "text-red-400"}>
                    {positive ? "+" : "-"}
                    {currency(Math.abs(b.pnl_usd))} ({positive ? "+" : "-"}
                    {pct(Math.abs(b.pnl_pct))})
                  </span>
                </span>
              );
            })}
          </div>
        )}
      </div>

      {session.error_message && (
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg border border-amber-900 bg-amber-950/30 p-4 text-xs text-amber-300">
          Last rebalance error (session is still active, will retry next cycle):{"\n"}
          {session.error_message}
        </pre>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="Equity" value={`$${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
        <MetricTile label="Cash" value={`$${account.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
        <MetricTile label="Buying power" value={`$${account.buying_power.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
        <MetricTile label="Open positions" value={String(account.positions.length)} />
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <h2 className="p-4 pb-3 text-sm font-medium text-slate-300">Current positions (live)</h2>
        {account.positions.length === 0 ? (
          <p className="p-4 pt-0 text-sm text-slate-500">No open positions.</p>
        ) : (
          <>
            <table className="w-full min-w-[480px] text-left text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-4 py-2 font-medium">Symbol</th>
                  <th className="px-4 py-2 font-medium">Qty</th>
                  <th className="px-4 py-2 font-medium">Avg entry</th>
                  <th className="px-4 py-2 font-medium">Market value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {positionsPage.pageItems.map((p) => (
                  <tr key={p.symbol}>
                    <td className="px-4 py-2 font-medium">{p.symbol}</td>
                    <td className="px-4 py-2">{num(p.qty)}</td>
                    <td className="px-4 py-2">${num(p.avg_entry_price)}</td>
                    <td className="px-4 py-2">${p.market_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Pagination page={positionsPage.page} totalPages={positionsPage.totalPages} onChange={positionsPage.setPage} />
          </>
        )}
      </div>

      {equityData.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-300">Equity curve (per rebalance)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={equityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} minTickGap={40} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} width={70} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Line type="monotone" dataKey="strategy" stroke={STRATEGY_COLOR} dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <h2 className="p-4 pb-3 text-sm font-medium text-slate-300">Order history</h2>
        {orders.length === 0 ? (
          <p className="p-4 pt-0 text-sm text-slate-500">No orders submitted yet.</p>
        ) : (
          <>
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-4 py-2 font-medium">Date</th>
                  <th className="px-4 py-2 font-medium">Symbol</th>
                  <th className="px-4 py-2 font-medium">Side</th>
                  <th className="px-4 py-2 font-medium">Notional</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 font-medium">Trigger</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {ordersPage.pageItems.map((o, i) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-slate-400">{o.date}</td>
                    <td className="px-4 py-2 font-medium">{o.symbol}</td>
                    <td className="px-4 py-2 capitalize">{o.side}</td>
                    <td className="px-4 py-2">${num(o.notional)}</td>
                    <td className="px-4 py-2 text-slate-400">{o.status}</td>
                    <td className="px-4 py-2">
                      {o.trigger === "anomaly" ? (
                        <span className="rounded-full bg-amber-900/40 px-2 py-0.5 text-xs font-medium text-amber-300">
                          anomaly
                        </span>
                      ) : (
                        <span className="text-xs text-slate-500">rebalance</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Pagination page={ordersPage.page} totalPages={ordersPage.totalPages} onChange={ordersPage.setPage} />
          </>
        )}
      </div>

      {aiUsage && (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <div className="flex flex-wrap items-center justify-between gap-1 p-4 pb-3">
            <h2 className="text-sm font-medium text-slate-300">AI sentiment call cost</h2>
            <span className="text-xs text-slate-500">estimated -- cross-check the Anthropic console for exact billing</span>
          </div>
          <div className="grid grid-cols-2 gap-3 px-4 pb-4 sm:grid-cols-4">
            <MetricTile label="Est. total spend" value={`$${aiUsage.total_estimated_cost_usd.toFixed(2)}`} />
            <MetricTile label="Calls made" value={String(aiUsage.total_calls)} />
            <MetricTile label="Input tokens" value={aiUsage.total_input_tokens.toLocaleString()} />
            <MetricTile label="Web searches" value={String(aiUsage.total_web_search_requests)} />
          </div>
          {aiUsage.events.length === 0 ? (
            <p className="p-4 pt-0 text-sm text-slate-500">No AI sentiment calls recorded yet.</p>
          ) : (
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-4 py-2 font-medium">Date</th>
                  <th className="px-4 py-2 font-medium">Input tok.</th>
                  <th className="px-4 py-2 font-medium">Output tok.</th>
                  <th className="px-4 py-2 font-medium">Searches</th>
                  <th className="px-4 py-2 font-medium">Est. cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {aiUsage.events.map((e, i) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-slate-400">{e.date}</td>
                    <td className="px-4 py-2">{e.input_tokens.toLocaleString()}</td>
                    <td className="px-4 py-2">{e.output_tokens.toLocaleString()}</td>
                    <td className="px-4 py-2">{e.web_search_requests}</td>
                    <td className="px-4 py-2">${e.estimated_cost_usd.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
