"use client";

import { useState } from "react";
import { Notice, PageHeader, pct } from "@/components/charts";
import { getHistory, type HistoryItem } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";

const PAGE = 20;

function when(iso: string) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function Row({ item }: { item: HistoryItem }) {
  const bad = item.label === "phishing";
  return (
    <tr className="border-t border-line">
      <td className="whitespace-nowrap px-3 py-2 text-muted">{when(item.created_at)}</td>
      <td className="px-3 py-2">
        <span className={`rounded px-2 py-0.5 font-mono text-xs font-bold uppercase ${bad ? "bg-danger/15 text-danger" : "bg-safe/15 text-safe"}`}>
          {bad ? "phishing" : "legitimate"}
        </span>
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="h-2 w-24 rounded bg-line/50" role="img" aria-label={`score ${pct(item.probability)}`}>
            <div className="h-full rounded" style={{ width: pct(item.probability, 0), background: bad ? "var(--danger)" : "var(--safe)" }} />
          </div>
          <span className="font-mono text-xs">{pct(item.probability)}</span>
        </div>
      </td>
      <td className="px-3 py-2 font-mono text-xs text-muted">{item.model_version}</td>
      <td className="px-3 py-2 font-mono text-xs text-muted" title={item.url_hash}>{item.url_hash.slice(0, 12)}…</td>
    </tr>
  );
}

const pagerButton =
  "rounded border border-line px-3 py-1 enabled:hover:border-accent enabled:hover:text-foreground disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-accent";

export default function HistoryPage() {
  const [offset, setOffset] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const { data, error, loading } = useFetch(() => getHistory(PAGE, offset), `${offset}:${refresh}`);

  const total = data?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE, total);

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6">
      <PageHeader kicker="history" title="Past predictions">
        Every URL checked through the detector is recorded with its verdict. The URL itself is never stored, only a keyed
        hash of it, so this list shows when something was checked and what the model said, not what was checked.
      </PageHeader>

      {error && <Notice kind="error">{error}</Notice>}
      {!error && !data && <Notice>Loading…</Notice>}
      {data && data.items.length === 0 && (
        <Notice>No predictions yet. Check a URL on the Detector page and it will show up here.</Notice>
      )}

      {data && data.items.length > 0 && (
        <section className="rounded-lg border border-line bg-panel">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] border-collapse text-left text-sm" aria-busy={loading}>
              <caption className="sr-only">Predictions, newest first</caption>
              <thead>
                <tr className="text-xs uppercase tracking-wider text-muted">
                  {["When", "Verdict", "Model score", "Model", "URL hash"].map((h) => (
                    <th key={h} scope="col" className="px-3 py-2 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className={loading ? "opacity-60" : undefined}>
                {data.items.map((i) => <Row key={i.id} item={i} />)}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-3 py-2 text-sm text-muted">
            <span aria-live="polite">{from}–{to} of {total.toLocaleString()}</span>
            <div className="flex gap-2">
              <button type="button" onClick={() => setRefresh((n) => n + 1)} className={pagerButton}>Refresh</button>
              <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} className={pagerButton}>
                Newer
              </button>
              <button type="button" disabled={to >= total} onClick={() => setOffset(offset + PAGE)} className={pagerButton}>
                Older
              </button>
            </div>
          </div>
        </section>
      )}
      <p className="mt-4 text-xs text-muted">
        The score is the model&apos;s estimate that the URL is phishing, not a certainty. The same URL always produces the
        same hash, so repeats can be spotted, but the URL cannot be recovered from the hash.
      </p>
    </main>
  );
}
