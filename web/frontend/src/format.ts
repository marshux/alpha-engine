export function pct(value: number | null, digits = 1): string {
  if (value === null || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null, digits = 2): string {
  if (value === null || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

export function currency(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "--";
  return value.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export function dateTime(value: string | null): string {
  if (!value) return "--";
  return new Date(value).toLocaleString();
}

export const STATUS_STYLES: Record<string, string> = {
  pending: "bg-slate-700 text-slate-200",
  running: "bg-amber-900 text-amber-300",
  completed: "bg-emerald-900 text-emerald-300",
  failed: "bg-red-900 text-red-300",
};
