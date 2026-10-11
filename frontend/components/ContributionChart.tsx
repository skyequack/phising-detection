import type { Contribution } from "@/lib/api";

export const fmt = (n: number) =>
  Number.isInteger(n) ? n.toLocaleString() : Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2);

export const pretty = (s: string) => s.replace(/_/g, " ");

// Diverging bars: right (red) pushed the probability up, left (green) pulled it down.
export function ContributionChart({ items }: { items: Contribution[] }) {
  const max = Math.max(0.01, ...items.map((i) => Math.abs(i.contribution)));
  return (
    <ul className="m-0 list-none space-y-2 p-0" aria-label="Top features by effect on the prediction">
      {items.map((i) => {
        const pct = (Math.abs(i.contribution) / max) * 50;
        const up = i.contribution > 0;
        return (
          <li key={i.feature} className="grid grid-cols-[minmax(0,9rem)_1fr_4.5rem] items-center gap-3 text-sm sm:grid-cols-[12rem_1fr_5rem]">
            <div className="min-w-0">
              <div className="truncate font-mono text-[13px]" title={i.feature}>{pretty(i.feature)}</div>
              <div className="truncate text-xs text-muted">
                {fmt(i.value)} <span aria-hidden>vs</span> typical legit {fmt(i.legit_median)}
              </div>
            </div>
            <div className="relative h-4 rounded bg-line/40">
              <div className="absolute inset-y-0 left-1/2 w-px bg-muted/60" />
              <div
                className="absolute inset-y-0 rounded"
                style={{
                  background: up ? "var(--danger)" : "var(--safe)",
                  width: `${pct}%`,
                  left: up ? "50%" : `${50 - pct}%`,
                }}
              />
            </div>
            <div className={`text-right font-mono text-xs ${up ? "text-danger" : "text-safe"}`}>
              {up ? "+" : "−"}{(Math.abs(i.contribution) * 100).toFixed(1)} pts
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export function GroupChart({ groups }: { groups: Record<string, number> }) {
  const entries = Object.entries(groups).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(0.01, ...entries.map(([, v]) => Math.abs(v)));
  return (
    <ul className="m-0 list-none space-y-2 p-0" aria-label="Effect by feature group">
      {entries.map(([g, v]) => (
        <li key={g} className="grid grid-cols-[6.5rem_1fr_4.5rem] items-center gap-3 text-sm">
          <span className="font-mono text-[13px]">{pretty(g)}</span>
          <div className="relative h-3 rounded bg-line/40">
            <div className="absolute inset-y-0 left-1/2 w-px bg-muted/60" />
            <div
              className="absolute inset-y-0 rounded"
              style={{
                background: v > 0 ? "var(--danger)" : "var(--safe)",
                width: `${(Math.abs(v) / max) * 50}%`,
                left: v > 0 ? "50%" : `${50 - (Math.abs(v) / max) * 50}%`,
              }}
            />
          </div>
          <span className={`text-right font-mono text-xs ${v > 0 ? "text-danger" : "text-safe"}`}>
            {v > 0 ? "+" : "−"}{(Math.abs(v) * 100).toFixed(1)} pts
          </span>
        </li>
      ))}
    </ul>
  );
}
