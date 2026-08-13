import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../api";
import { buildDrawdownChartData, buildEquityChartData } from "../chartData";
import { ProgressBar } from "../components/ProgressBar";
import { dateTime, num, pct, STATUS_STYLES } from "../format";
import type { RunDetail as RunDetailType } from "../types";

const POLL_INTERVAL_MS = 4000;
const BENCHMARK_COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#f472b6"];
const STRATEGY_COLOR = "#34d399";

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}

export function RunDetail() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<RunDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visibilityPending, setVisibilityPending] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const toggleVisibility = async () => {
    if (!run) return;
    setVisibilityPending(true);
    try {
      const updated = await api.setRunVisibility(run.id, !run.is_public);
      setRun({ ...run, is_public: updated.is_public });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setVisibilityPending(false);
    }
  };

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    const load = () => {
      api
        .getRun(Number(id))
        .then((data) => {
          if (cancelled) return;
          setRun(data);
          setError(null);
          if (data.status === "pending" || data.status === "running") {
            timerRef.current = setTimeout(load, POLL_INTERVAL_MS);
          }
        })
        .catch((e) => !cancelled && setError(e instanceof ApiError ? e.message : String(e)));
    };
    load();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [id]);

  if (error) return <p className="text-red-400">Failed to load run: {error}</p>;
  if (run === null) return <p className="text-slate-400">Loading...</p>;

  const benchmarkSymbols = Array.from(new Set(run.benchmark_points.map((p) => p.symbol)));
  const equityData = buildEquityChartData(run);
  const drawdownData = buildDrawdownChartData(run);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold">
            Run #{run.id} <span className="text-slate-500">-- {run.run_type}</span>
          </h1>
          <p className="text-sm text-slate-400">
            {run.start_date} → {run.end_date} -- created {dateTime(run.created_at)}
          </p>
        </div>
        <div className="text-right">
          <span className={`rounded px-2 py-1 text-xs font-medium ${STATUS_STYLES[run.status]}`}>
            {run.status}
          </span>
          {run.status === "running" && run.progress_total !== null && (
            <div className="mt-2">
              <ProgressBar current={run.progress_current} total={run.progress_total} />
            </div>
          )}
          {run.status === "completed" && (
            <button
              onClick={toggleVisibility}
              disabled={visibilityPending}
              className={`mt-2 block w-full rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50 ${
                run.is_public
                  ? "bg-emerald-900 text-emerald-300 hover:bg-emerald-800"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {run.is_public ? "Public -- click to unpublish" : "Private -- click to publish"}
            </button>
          )}
        </div>
      </div>

      {run.status === "failed" && run.error_message && (
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-red-900 bg-red-950/40 p-4 text-xs text-red-300">
          {run.error_message}
        </pre>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="CAGR" value={pct(run.cagr)} />
        <MetricTile label="Sharpe" value={num(run.sharpe)} />
        <MetricTile label="Sortino" value={num(run.sortino)} />
        <MetricTile label="Calmar" value={num(run.calmar)} />
        <MetricTile label="Max drawdown" value={pct(run.max_drawdown)} />
        <MetricTile label="Win rate" value={pct(run.win_rate)} />
        <MetricTile label="Alpha vs benchmark" value={pct(run.alpha_vs_primary_benchmark)} />
      </div>

      {equityData.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-300">Equity curve</h2>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={equityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} minTickGap={40} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} width={70} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Legend />
              <Line type="monotone" dataKey="strategy" stroke={STRATEGY_COLOR} dot={false} strokeWidth={2} />
              {benchmarkSymbols.map((symbol, i) => (
                <Line
                  key={symbol}
                  type="monotone"
                  dataKey={symbol}
                  stroke={BENCHMARK_COLORS[i % BENCHMARK_COLORS.length]}
                  dot={false}
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {drawdownData.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-300">Drawdown</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={drawdownData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} minTickGap={40} />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                width={60}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }}
                formatter={(v) => pct(typeof v === "number" ? v : null)}
              />
              <Line type="monotone" dataKey="drawdown" stroke="#f87171" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {run.tearsheet_path && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-300">Tear sheet</h2>
          <img src={api.tearsheetUrl(run.id)} alt="Tear sheet" className="w-full rounded" />
        </div>
      )}

      {run.walkforward_windows.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <h2 className="p-4 pb-0 text-sm font-medium text-slate-300">Walk-forward windows</h2>
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-slate-900 text-slate-400">
              <tr>
                <th className="px-4 py-2 font-medium">#</th>
                <th className="px-4 py-2 font-medium">Train</th>
                <th className="px-4 py-2 font-medium">Test</th>
                <th className="px-4 py-2 font-medium">CAGR</th>
                <th className="px-4 py-2 font-medium">Sharpe</th>
                <th className="px-4 py-2 font-medium">Max DD</th>
                <th className="px-4 py-2 font-medium">Winning overrides</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {run.walkforward_windows.map((w) => (
                <tr key={w.window_index}>
                  <td className="px-4 py-2">{w.window_index}</td>
                  <td className="px-4 py-2 text-slate-400">
                    {w.train_start} → {w.train_end}
                  </td>
                  <td className="px-4 py-2 text-slate-400">
                    {w.test_start} → {w.test_end}
                  </td>
                  <td className="px-4 py-2">{pct(w.cagr)}</td>
                  <td className="px-4 py-2">{num(w.sharpe)}</td>
                  <td className="px-4 py-2">{pct(w.max_drawdown)}</td>
                  <td className="max-w-xs truncate px-4 py-2 font-mono text-xs text-slate-400">
                    {w.winning_overrides_json}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
