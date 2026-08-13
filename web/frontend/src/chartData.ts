import type { BenchmarkPoint, EquityPoint } from "./types";

interface HasCurves {
  equity_points: EquityPoint[];
  benchmark_points: BenchmarkPoint[];
}

export function buildEquityChartData(run: HasCurves) {
  const byDate = new Map<string, Record<string, number>>();
  for (const p of run.equity_points) {
    byDate.set(p.date, { ...(byDate.get(p.date) ?? {}), strategy: p.equity });
  }
  for (const p of run.benchmark_points) {
    byDate.set(p.date, { ...(byDate.get(p.date) ?? {}), [p.symbol]: p.equity });
  }
  return Array.from(byDate.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, values]) => ({ date, ...values }));
}

export function buildDrawdownChartData(run: Pick<HasCurves, "equity_points">) {
  let peak = -Infinity;
  return [...run.equity_points]
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((p) => {
      peak = Math.max(peak, p.equity);
      return { date: p.date, drawdown: peak > 0 ? p.equity / peak - 1 : 0 };
    });
}
