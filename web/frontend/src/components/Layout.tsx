import { Link, Outlet } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";

export function Layout() {
  const { me, refresh } = useAuth();

  const handleLogout = async () => {
    await api.logout();
    refresh();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="text-lg font-semibold tracking-tight text-slate-100">
            alpha-engine
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link to="/" className="text-slate-300 hover:text-white">
              Runs
            </Link>
            <Link
              to="/runs/new"
              className="rounded-md bg-emerald-600 px-3 py-1.5 font-medium text-white hover:bg-emerald-500"
            >
              New run
            </Link>
            {me?.username && (
              <span className="flex items-center gap-3 border-l border-slate-700 pl-4 text-slate-400">
                {me.username}
                <button onClick={handleLogout} className="text-slate-400 hover:text-white">
                  Sign out
                </button>
              </span>
            )}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
