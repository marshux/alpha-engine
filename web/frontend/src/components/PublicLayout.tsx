import { useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";
import { api } from "../api";

export function PublicLayout() {
  const [delayDays, setDelayDays] = useState<number | null>(null);

  useEffect(() => {
    api
      .publicMeta()
      .then((meta) => setDelayDays(meta.delay_days))
      .catch(() => setDelayDays(null));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/public" className="text-lg font-semibold tracking-tight text-slate-100">
            alpha-engine
          </Link>
          <Link to="/login" className="text-sm text-slate-400 hover:text-white">
            Owner sign in
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <p className="mb-6 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-2 text-sm text-slate-400">
          Public read-only view -- aggregate returns only{delayDays !== null ? `, delayed by ${delayDays} days` : ""}
          . No positions, trades, or strategy details are shown.
        </p>
        <Outlet />
      </main>
    </div>
  );
}
