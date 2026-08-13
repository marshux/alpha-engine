export function ProgressBar({ current, total }: { current: number | null; total: number | null }) {
  if (current === null || total === null || total === 0) {
    return <span className="text-xs text-slate-500">starting...</span>;
  }
  const fraction = Math.min(1, current / total);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full bg-amber-500 transition-all" style={{ width: `${fraction * 100}%` }} />
      </div>
      <span className="text-xs text-slate-400">
        {current}/{total}
      </span>
    </div>
  );
}
