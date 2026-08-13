// Mirrors web/backend/app/schemas.py -- keep in sync by hand (no codegen
// yet; the backend is small enough that this hasn't been worth automating).

export type RunType = "backtest" | "walkforward";
export type RunStatus = "pending" | "running" | "completed" | "failed";

export interface RunSummary {
  id: number;
  run_type: RunType;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  start_date: string;
  end_date: string;
  cagr: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  final_equity: number | null;
  progress_current: number | null;
  progress_total: number | null;
  is_public: boolean;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface BenchmarkPoint {
  symbol: string;
  date: string;
  equity: number;
}

export interface WalkForwardWindow {
  window_index: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  winning_overrides_json: string;
  cagr: number | null;
  sharpe: number | null;
  sortino: number | null;
  calmar: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
}

export interface RunDetail extends RunSummary {
  error_message: string | null;
  config_json: string;
  universe_json: string;
  sortino: number | null;
  calmar: number | null;
  win_rate: number | null;
  alpha_vs_primary_benchmark: number | null;
  tearsheet_path: string | null;
  equity_points: EquityPoint[];
  benchmark_points: BenchmarkPoint[];
  walkforward_windows: WalkForwardWindow[];
}

export interface RunCreateRequest {
  run_type: RunType;
  start_date: string;
  end_date: string;
  symbols?: string[] | null;
  fundamentals?: boolean;
  config_overrides?: Record<string, unknown> | null;
  train_years?: number | null;
  test_years?: number | null;
  step_years?: number | null;
  param_grid?: Record<string, unknown[]> | null;
}

export interface AuthMe {
  authenticated: boolean;
  username: string | null;
}

// Mirrors PublicRunSummary/PublicRunDetail in app/schemas.py -- deliberately
// missing config_json/universe_json/tearsheet_path/walkforward_windows,
// which the backend never sends to unauthenticated callers.
export interface PublicRunSummary {
  id: number;
  run_type: RunType;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  start_date: string;
  end_date: string;
  cagr: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  final_equity: number | null;
}

export interface PublicRunDetail extends PublicRunSummary {
  equity_points: EquityPoint[];
  benchmark_points: BenchmarkPoint[];
}
