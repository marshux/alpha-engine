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
import { num } from "../format";
import type { LiveAccount, RunDetail } from "../types";

const POSITIONS_POLL_MS = 10_000;
const STRATEGY_COLOR = "#34d399";

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

  useEffect(() => {
    loadSession().catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    loadPositions();
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

  if (error) return <p className="text-red-400">Error: {error}</p>;
  if (session === null || account === null) return <p className="text-slate-400">Loading...</p>;

  const equityData = session.equity_points.length > 0 ? buildEquityChartData(session) : [];

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
            onClick={handleFlatten}
            disabled={actionPending !== null}
            className="rounded-md bg-red-900 px-3 py-1.5 text-sm font-medium text-red-200 hover:bg-red-800 disabled:opacity-50"
          >
            {actionPending === "flatten" ? "Flattening..." : "Flatten to cash"}
          </button>
        </div>
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
        <h2 className="p-4 pb-0 text-sm font-medium text-slate-300">Current positions (live)</h2>
        {account.positions.length === 0 ? (
          <p className="p-4 text-sm text-slate-500">No open positions.</p>
        ) : (
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
              {account.positions.map((p) => (
                <tr key={p.symbol}>
                  <td className="px-4 py-2 font-medium">{p.symbol}</td>
                  <td className="px-4 py-2">{num(p.qty)}</td>
                  <td className="px-4 py-2">${num(p.avg_entry_price)}</td>
                  <td className="px-4 py-2">${p.market_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
        <h2 className="p-4 pb-0 text-sm font-medium text-slate-300">Order history</h2>
        {session.live_orders.length === 0 ? (
          <p className="p-4 text-sm text-slate-500">No orders submitted yet.</p>
        ) : (
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead className="bg-slate-900 text-slate-400">
              <tr>
                <th className="px-4 py-2 font-medium">Date</th>
                <th className="px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Side</th>
                <th className="px-4 py-2 font-medium">Notional</th>
                <th className="px-4 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {[...session.live_orders].reverse().map((o, i) => (
                <tr key={i}>
                  <td className="px-4 py-2 text-slate-400">{o.date}</td>
                  <td className="px-4 py-2 font-medium">{o.symbol}</td>
                  <td className="px-4 py-2 capitalize">{o.side}</td>
                  <td className="px-4 py-2">${num(o.notional)}</td>
                  <td className="px-4 py-2 text-slate-400">{o.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
