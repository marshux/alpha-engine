import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { dateTime, num, pct } from "../format";
import type { PublicRunSummary } from "../types";

export function PublicRunHistory() {
  const [runs, setRuns] = useState<PublicRunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listPublicRuns()
      .then(setRuns)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="text-red-400">Failed to load: {error}</p>;
  if (runs === null) return <p className="text-slate-400">Loading...</p>;
  if (runs.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center text-slate-400">
        Nothing published yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="bg-slate-900 text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">Type</th>
            <th className="px-4 py-3 font-medium">Range</th>
            <th className="px-4 py-3 font-medium">CAGR</th>
            <th className="px-4 py-3 font-medium">Sharpe</th>
            <th className="px-4 py-3 font-medium">Max DD</th>
            <th className="px-4 py-3 font-medium">Published</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 bg-slate-950">
          {runs.map((run) => (
            <tr key={run.id} className="hover:bg-slate-900/60">
              <td className="px-4 py-3 capitalize">
                <Link to={`/public/runs/${run.id}`} className="text-emerald-400 hover:underline">
                  {run.run_type}
                </Link>
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
