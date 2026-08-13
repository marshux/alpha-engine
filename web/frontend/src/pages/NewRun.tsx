import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
import type { RunCreateRequest, RunType } from "../types";

const today = new Date().toISOString().slice(0, 10);
const sixYearsAgo = new Date(new Date().setFullYear(new Date().getFullYear() - 6))
  .toISOString()
  .slice(0, 10);

export function NewRun() {
  const navigate = useNavigate();
  const [runType, setRunType] = useState<RunType>("backtest");
  const [startDate, setStartDate] = useState(sixYearsAgo);
  const [endDate, setEndDate] = useState(today);
  const [symbols, setSymbols] = useState("");
  const [fundamentals, setFundamentals] = useState(false);
  const [trainYears, setTrainYears] = useState(3);
  const [testYears, setTestYears] = useState(1);
  const [stepYears, setStepYears] = useState(1);
  const [useTuningGrid, setUseTuningGrid] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const payload: RunCreateRequest = {
        run_type: runType,
        start_date: startDate,
        end_date: endDate,
        symbols: symbols.trim() ? symbols.split(",").map((s) => s.trim().toUpperCase()) : null,
        fundamentals,
      };
      if (runType === "walkforward") {
        payload.train_years = trainYears;
        payload.test_years = testYears;
        payload.step_years = stepYears;
        // The actual weight/sizing profiles live in config/tuning_grid.yaml
        // server-side; this dashboard doesn't (yet) expose an editor for
        // arbitrary param grids, just a one-click "use the default one".
        if (useTuningGrid) {
          payload.param_grid = await api.defaultTuningGrid();
        }
      }
      const run = await api.createRun(payload);
      navigate(`/runs/${run.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setSubmitting(false);
    }
  };

  const inputClass =
    "w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none";
  const labelClass = "mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500";

  return (
    <div className="max-w-xl">
      <h1 className="mb-6 text-2xl font-semibold">New run</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div>
          <label className={labelClass}>Run type</label>
          <div className="flex gap-2">
            {(["backtest", "walkforward"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setRunType(t)}
                className={`rounded-md px-4 py-2 text-sm font-medium capitalize ${
                  runType === t ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-300"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelClass}>Start date</label>
            <input
              type="date"
              className={inputClass}
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              required
            />
          </div>
          <div>
            <label className={labelClass}>End date</label>
            <input
              type="date"
              className={inputClass}
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              required
            />
          </div>
        </div>

        <div>
          <label className={labelClass}>Symbols (optional -- comma separated, blank = full universe)</label>
          <input
            type="text"
            className={inputClass}
            placeholder="AAPL, MSFT, NVDA"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={fundamentals}
            onChange={(e) => setFundamentals(e.target.checked)}
            className="rounded border-slate-700 bg-slate-900"
          />
          Include fundamentals (enables value/quality signals; slower fetch)
        </label>

        {runType === "walkforward" && (
          <>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className={labelClass}>Train years</label>
                <input
                  type="number"
                  step="0.25"
                  min="0.25"
                  className={inputClass}
                  value={trainYears}
                  onChange={(e) => setTrainYears(Number(e.target.value))}
                />
              </div>
              <div>
                <label className={labelClass}>Test years</label>
                <input
                  type="number"
                  step="0.25"
                  min="0.25"
                  className={inputClass}
                  value={testYears}
                  onChange={(e) => setTestYears(Number(e.target.value))}
                />
              </div>
              <div>
                <label className={labelClass}>Step years</label>
                <input
                  type="number"
                  step="0.25"
                  min="0.25"
                  className={inputClass}
                  value={stepYears}
                  onChange={(e) => setStepYears(Number(e.target.value))}
                />
              </div>
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={useTuningGrid}
                onChange={(e) => setUseTuningGrid(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900"
              />
              Grid-search config/tuning_grid.yaml profiles on each train window (much slower)
            </label>
          </>
        )}

        {error && (
          <p className="rounded-md border border-red-900 bg-red-950/40 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="rounded-md bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {submitting ? "Starting..." : "Start run"}
        </button>
      </form>
    </div>
  );
}
