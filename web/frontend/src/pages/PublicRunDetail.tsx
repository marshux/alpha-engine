import { useEffect, useState } from "react";
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
import { pct } from "../format";
import type { PublicRunDetail as PublicRunDetailType } from "../types";

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

export function PublicRunDetail() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<PublicRunDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    api
      .getPublicRun(Number(id))
      .then(setRun)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [id]);

  if (error) return <p className="text-red-400">Failed to load: {error}</p>;
  if (run === null) return <p className="text-slate-400">Loading...</p>;

  const benchmarkSymbols = Array.from(new Set(run.benchmark_points.map((p) => p.symbol)));
  const equityData = buildEquityChartData(run);
  const drawdownData = buildDrawdownChartData(run);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold capitalize">
        {run.run_type} <span className="text-slate-500">-- {run.start_date} → {run.end_date}</span>
      </h1>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="CAGR" value={pct(run.cagr)} />
        <MetricTile label="Sharpe" value={run.sharpe?.toFixed(2) ?? "--"} />
        <MetricTile label="Max drawdown" value={pct(run.max_drawdown)} />
        <MetricTile label="Win rate" value={pct(run.win_rate)} />
      </div>

      {equityData.length > 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-300">Portfolio value</h2>
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
      ) : (
        <p className="text-sm text-slate-500">No data old enough to display yet.</p>
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
    </div>
  );
}
