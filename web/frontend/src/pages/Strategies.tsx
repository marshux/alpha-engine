import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api";
import { currency, pct } from "../format";
import type { StrategySummary } from "../types";

const POLL_INTERVAL_MS = 15_000;

const RISK_TIER_STYLES: Record<string, string> = {
  traditional: "bg-sky-900 text-sky-300",
  aggressive: "bg-red-900 text-red-300",
  systematic: "bg-violet-900 text-violet-300",
};

function RiskTierBadge({ tier }: { tier: string }) {
  const style = RISK_TIER_STYLES[tier] ?? "bg-slate-700 text-slate-200";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${style}`}>{tier}</span>;
}

export function Strategies() {
  const [strategies, setStrategies] = useState<StrategySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const load = () => {
      api
        .listStrategies()
        .then(setStrategies)
        .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    };
    load();
    timerRef.current = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  if (error) return <p className="text-red-400">Error: {error}</p>;
  if (strategies === null) return <p className="text-slate-400">Loading...</p>;

  if (strategies.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center text-slate-400">
        No strategies configured yet.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Live strategies</h1>
        <p className="text-sm text-slate-400">Each strategy trades its own independent Alpaca paper account.</p>
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-slate-900 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Strategy</th>
              <th className="px-4 py-3 font-medium">Risk tier</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium">Equity</th>
              <th className="px-4 py-3 font-medium">Total P/L</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 bg-slate-950">
            {strategies.map((s) => {
              const positive = (s.total_pnl_usd ?? 0) >= 0;
              return (
                <tr key={s.id} className="hover:bg-slate-900/60">
                  <td className="px-4 py-3">
                    <Link to={`/live/${s.id}`} className="font-medium text-emerald-400 hover:underline">
                      {s.display_name}
                    </Link>
                    <div className="text-xs text-slate-500">{s.description}</div>
                  </td>
                  <td className="px-4 py-3">
                    <RiskTierBadge tier={s.risk_tier} />
                    {!s.enabled && (
                      <span className="ml-2 rounded-full bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-400">
                        not yet scheduled
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-400">{s.status ?? "no runs yet"}</td>
                  <td className="px-4 py-3">{s.equity !== null ? currency(s.equity) : "--"}</td>
                  <td className="px-4 py-3">
                    {s.total_pnl_usd !== null ? (
                      <span className={positive ? "text-emerald-400" : "text-red-400"}>
                        {positive ? "+" : "-"}
                        {currency(Math.abs(s.total_pnl_usd))} (
                        {positive ? "+" : "-"}
                        {pct(Math.abs(s.total_pnl_pct ?? 0))})
                      </span>
                    ) : (
                      <span className="text-slate-500">--</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
