// Thin fetch wrapper. Same-origin in both dev (vite.config.ts proxies
// /api and /auth to the backend) and production (the backend serves this
// app's built static files itself) -- so no CORS setup is needed anywhere,
// and the session cookie just works via `credentials: "include"`.
import type {
  AiUsageSummary,
  AuthMe,
  LiveAccount,
  PublicRunDetail,
  PublicRunSummary,
  RunCreateRequest,
  RunDetail,
  RunSummary,
} from "./types";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "include",
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  me: () => request<AuthMe>("/auth/me"),
  logout: () => request<{ status: string }>("/auth/logout", { method: "POST" }),

  listRuns: () => request<RunSummary[]>("/api/runs"),
  getRun: (id: number) => request<RunDetail>(`/api/runs/${id}`),
  createRun: (payload: RunCreateRequest) =>
    request<RunSummary>("/api/runs", { method: "POST", body: JSON.stringify(payload) }),
  tearsheetUrl: (id: number) => `/api/runs/${id}/tearsheet`,
  defaultTuningGrid: () => request<Record<string, unknown[]>>("/api/runs/tuning-grid/default"),
  setRunVisibility: (id: number, isPublic: boolean) =>
    request<RunSummary>(`/api/runs/${id}`, { method: "PATCH", body: JSON.stringify({ is_public: isPublic }) }),

  listPublicRuns: () => request<PublicRunSummary[]>("/api/public/runs"),
  getPublicRun: (id: number) => request<PublicRunDetail>(`/api/public/runs/${id}`),
  publicMeta: () => request<{ delay_days: number }>("/api/public/meta"),

  getLivePositions: () => request<LiveAccount>("/api/live/positions"),
  getAiUsage: () => request<AiUsageSummary>("/api/live/ai-usage"),
  startPaperTrading: () => request<RunSummary>("/api/live/start", { method: "POST" }),
  rebalanceNow: () => request<RunSummary>("/api/live/rebalance-now", { method: "POST" }),
  checkAnomaliesNow: () => request<RunSummary>("/api/live/anomaly-check-now", { method: "POST" }),
  flattenToCash: () => request<{ status: string }>("/api/live/flatten", { method: "POST" }),
};

export { ApiError };
