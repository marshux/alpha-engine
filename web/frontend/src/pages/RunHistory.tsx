import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ProgressBar } from "../components/ProgressBar";
import { dateTime, num, pct, STATUS_STYLES } from "../format";
import type { RunSummary } from "../types";

const POLL_INTERVAL_MS = 4000;

export function RunHistory() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = () => {
      api
        .listRuns()
        .then((data) => {
          if (cancelled) return;
          setRuns(data);
          setError(null);
          const stillActive = data.some((r) => r.status === "pending" || r.status === "running");
          timerRef.current = setTimeout(load, stillActive ? POLL_INTERVAL_MS : POLL_INTERVAL_MS * 4);
        })
        .catch((e) => !cancelled && setError(String(e)));
    };
    load();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (error) return <p className="text-red-400">Failed to load runs: {error}</p>;
  if (runs === null) return <p className="text-slate-400">Loading...</p>;
  if (runs.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center text-slate-400">
        No runs yet.{" "}
        <Link to="/runs/new" className="text-emerald-400 hover:underline">
          Start one
        </Link>
        .
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-slate-900 text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">ID</th>
            <th className="px-4 py-3 font-medium">Type</th>
            <th className="px-4 py-3 font-medium">Status</th>
            <th className="px-4 py-3 font-medium">Range</th>
            <th className="px-4 py-3 font-medium">CAGR</th>
            <th className="px-4 py-3 font-medium">Sharpe</th>
            <th className="px-4 py-3 font-medium">Max DD</th>
            <th className="px-4 py-3 font-medium">Created</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 bg-slate-950">
          {runs.map((run) => (
            <tr key={run.id} className="hover:bg-slate-900/60">
              <td className="px-4 py-3">
                <Link to={`/runs/${run.id}`} className="text-emerald-400 hover:underline">
                  #{run.id}
                </Link>
              </td>
              <td className="px-4 py-3 capitalize">{run.run_type}</td>
              <td className="px-4 py-3">
                <span className={`rounded px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[run.status]}`}>
                  {run.status}
                </span>
                {run.status === "running" && run.progress_total !== null && (
                  <div className="mt-1">
                    <ProgressBar current={run.progress_current} total={run.progress_total} />
                  </div>
                )}
              </td>
              <td className="px-4 py-3 text-slate-400">
                {run.start_date} → {run.end_date}
              </td>
              <td className="px-4 py-3">{pct(run.cagr)}</td>
              <td className="px-4 py-3">{num(run.sharpe)}</td>
              <td className="px-4 py-3">{pct(run.max_drawdown)}</td>
              <td className="px-4 py-3 text-slate-400">{dateTime(run.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
