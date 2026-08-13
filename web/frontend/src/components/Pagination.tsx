// Client-side pagination -- every paginated list on this dashboard (live
// positions, order history) is already fetched in full from its API
// response, so there's no server-side page param to thread through; this
// just slices an in-memory array. `usePagination` clamps automatically
// when the underlying list shrinks (e.g. a position closes) so a stale
// page number never points past the end.
import { useState } from "react";

export function usePagination<T>(items: T[], pageSize: number) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  return {
    pageItems: items.slice(start, start + pageSize),
    page: safePage,
    totalPages,
    setPage,
  };
}

export function Pagination({
  page,
  totalPages,
  onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (page: number) => void;
}) {
  if (totalPages <= 1) return null;
  return (
    <div className="flex items-center justify-between border-t border-slate-800 px-4 py-2 text-xs text-slate-400">
      <button
        onClick={() => onChange(page - 1)}
        disabled={page <= 1}
        className="rounded px-2 py-1 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-30 disabled:hover:bg-transparent"
      >
        Prev
      </button>
      <span>
        Page {page} of {totalPages}
      </span>
      <button
        onClick={() => onChange(page + 1)}
        disabled={page >= totalPages}
        className="rounded px-2 py-1 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-30 disabled:hover:bg-transparent"
      >
        Next
      </button>
    </div>
  );
}
