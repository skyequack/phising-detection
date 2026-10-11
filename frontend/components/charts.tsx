import type { Confusion, SweepRow } from "@/lib/api";
import { pretty } from "./ContributionChart";

export function ThresholdSweep({ rows, chosen }: { rows: SweepRow[]; chosen: number }) {
  const W = 360;
  const H = 180;
  const pad = { l: 34, r: 10, t: 10, b: 24 };
  const xs = rows.map((r) => r.threshold);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const X = (t: number) => pad.l + ((t - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const line = (k: "precision" | "recall" | "f1") => rows.map((r) => `${X(r.threshold)},${Y(r[k])}`).join(" ");
  const series = [
    ["precision", "var(--accent)"],
    ["recall", "var(--danger)"],
    ["f1", "var(--safe)"],
  ] as const;
  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label="Precision, recall and F1 across decision thresholds, with the chosen threshold marked">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--panel-border)" />
            <text x={pad.l - 6} y={Y(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted)">{v}</text>
          </g>
        ))}
        {[x0, (x0 + x1) / 2, x1].map((t) => (
          <text key={t} x={X(t)} y={H - 8} textAnchor="middle" fontSize="9" fill="var(--muted)">{t.toFixed(2)}</text>
        ))}
        {series.map(([k, c]) => (
          <polyline key={k} points={line(k)} fill="none" stroke={c} strokeWidth="2" />
        ))}
        <line x1={X(chosen)} x2={X(chosen)} y1={pad.t} y2={H - pad.b} stroke="var(--foreground)" strokeDasharray="4 3" />
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 text-xs text-muted">
        {series.map(([k, c]) => (
          <span key={k}><span style={{ color: c }}>●</span> {k}</span>
        ))}
        <span>┆ chosen threshold {chosen.toFixed(2)}</span>
      </figcaption>
    </figure>
  );
}

export function ConfusionMatrix({ c }: { c: Confusion }) {
  const cells = [
    ["True legit", c.tn, "var(--safe)"],
    ["False alarm", c.fp, "var(--accent)"],
    ["Missed phishing", c.fn, "var(--danger)"],
    ["Caught phishing", c.tp, "var(--safe)"],
  ] as const;
  const max = Math.max(...cells.map((x) => x[1]));
  return (
    <div className="grid grid-cols-2 gap-1.5">
      {cells.map(([name, v, color]) => (
        <div key={name} className="rounded border border-line p-3"
          style={{ background: `color-mix(in srgb, ${color} ${Math.round(8 + (v / max) * 30)}%, transparent)` }}>
          <div className="font-mono text-lg">{v.toLocaleString()}</div>
          <div className="text-xs text-muted">{name}</div>
        </div>
      ))}
    </div>
  );
}

export function Importance({ groups }: { groups: Record<string, number> }) {
  const entries = Object.entries(groups).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map((e) => e[1]), 1e-9);
  return (
    <ul className="m-0 list-none space-y-2 p-0">
      {entries.map(([g, v]) => (
        <li key={g} className="grid grid-cols-[6.5rem_1fr_3rem] items-center gap-3 text-sm">
          <span className="font-mono text-[13px]">{pretty(g)}</span>
          <div className="h-3 rounded bg-line/40">
            <div className="h-full rounded bg-accent" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono text-xs text-muted">{(v * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}

export const Card = ({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) => (
  <section className="rounded-lg border border-line bg-panel p-5">
    <h3 className="m-0 text-sm font-semibold uppercase tracking-wider text-muted">{title}</h3>
    {note && <p className="mb-3 mt-1 text-xs text-muted">{note}</p>}
    <div className={note ? "" : "mt-3"}>{children}</div>
  </section>
);


export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export const PALETTE = ["var(--accent)", "var(--danger)", "var(--safe)", "#c084fc", "#fbbf24"];

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded border border-line bg-background/40 p-3">
      <div className="font-mono text-xl">{value}</div>
      <div className="text-xs text-muted">{label}</div>
      {note && <div className="mt-1 text-[11px] text-muted/80">{note}</div>}
    </div>
  );
}

export type Series = { name: string; color: string; points: [number, number][]; dashed?: boolean; dots?: boolean };

/** Square 0..1 line chart (ROC, precision-recall, calibration), with an optional y = x reference line. */
export function LineChart({ series, xLabel, yLabel, diagonal = false, label }: {
  series: Series[]; xLabel: string; yLabel: string; diagonal?: boolean; label: string;
}) {
  const W = 320;
  const H = 280;
  const pad = { l: 38, r: 10, t: 10, b: 36 };
  const X = (v: number) => pad.l + v * (W - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--panel-border)" />
            <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--panel-border)" />
            <text x={pad.l - 6} y={Y(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted)">{v}</text>
            <text x={X(v)} y={H - pad.b + 13} textAnchor="middle" fontSize="9" fill="var(--muted)">{v}</text>
          </g>
        ))}
        {diagonal && <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke="var(--muted)" strokeDasharray="4 3" />}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "5 3" : undefined}
              points={s.points.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")} />
            {s.dots && s.points.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r="3" fill={s.color} />)}
          </g>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--muted)">{xLabel}</text>
        <text transform={`translate(10 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize="10" fill="var(--muted)">
          {yLabel}
        </text>
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 text-xs text-muted">
        {series.map((s) => (
          <span key={s.name}><span style={{ color: s.color }}>●</span> {s.name}</span>
        ))}
      </figcaption>
    </figure>
  );
}

/** Horizontal bars that can go left (worse, red) or right (better, green) of a centre line. */
export function DeltaBars({ rows, unit = "pts" }: { rows: { label: string; value: number; hint?: string }[]; unit?: string }) {
  const max = Math.max(0.001, ...rows.map((r) => Math.abs(r.value)));
  return (
    <ul className="m-0 list-none space-y-2 p-0">
      {rows.map((r) => {
        const w = (Math.abs(r.value) / max) * 50;
        const worse = r.value < 0;
        return (
          <li key={r.label} className="grid grid-cols-[minmax(0,8.5rem)_1fr_4.5rem] items-center gap-3 text-sm sm:grid-cols-[12rem_1fr_5rem]">
            <span className="truncate font-mono text-[13px]" title={r.hint ?? r.label}>{r.label}</span>
            <div className="relative h-3 rounded bg-line/40">
              <div className="absolute inset-y-0 left-1/2 w-px bg-muted/60" />
              <div className="absolute inset-y-0 rounded"
                style={{ background: worse ? "var(--danger)" : "var(--safe)", width: `${w}%`, left: worse ? `${50 - w}%` : "50%" }} />
            </div>
            <span className={`text-right font-mono text-xs ${worse ? "text-danger" : "text-safe"}`}>
              {r.value > 0 ? "+" : r.value < 0 ? "−" : ""}{(Math.abs(r.value) * 100).toFixed(1)} {unit}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function PageHeader({ kicker, title, children }: { kicker: string; title: string; children?: React.ReactNode }) {
  return (
    <header className="mb-8">
      <p className="m-0 font-mono text-xs uppercase tracking-[0.25em] text-accent">{kicker}</p>
      <h1 className="mb-2 mt-2 text-3xl font-bold tracking-tight">{title}</h1>
      {children && <div className="max-w-2xl text-muted">{children}</div>}
    </header>
  );
}

export function Notice({ kind = "info", children }: { kind?: "info" | "error"; children: React.ReactNode }) {
  const tone = kind === "error" ? "border-danger/50 bg-danger/10 text-danger" : "border-line bg-panel text-muted";
  return <p role={kind === "error" ? "alert" : "status"} className={`rounded-lg border px-4 py-3 text-sm ${tone}`}>{children}</p>;
}
