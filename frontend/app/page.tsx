"use client";

import { useEffect, useState } from "react";
import Gauge from "@/components/Gauge";
import { ContributionChart, GroupChart } from "@/components/ContributionChart";
import ModelPanel from "@/components/ModelPanel";
import {
  MAX_URL_LENGTH, getMetrics, getModel, predict,
  type Metrics, type ModelInfo, type Prediction,
} from "@/lib/api";

const EXAMPLES = [
  "https://en.wikipedia.org/wiki/Cat",
  "http://paypal-secure-login.verify-account.example.ru/signin?id=9f8e7d6c5b4a",
  "http://192.168.4.7/update/confirm-password.php",
];

function Verdict({ r }: { r: Prediction }) {
  const bad = r.label === "phishing";
  const parts = r.url_parts;
  return (
    <section className="rise rounded-lg border bg-panel p-5" style={{ borderColor: bad ? "var(--danger)" : "var(--safe)" }}
      aria-live="polite">
      <div className="grid items-center gap-6 md:grid-cols-[auto_1fr]">
        <Gauge probability={r.probability} threshold={r.threshold} />
        <div>
          <div className={`inline-block rounded px-3 py-1 font-mono text-sm font-bold uppercase tracking-widest ${bad ? "bg-danger/15 text-danger" : "bg-safe/15 text-safe"}`}>
            {bad ? "Likely phishing" : "Looks legitimate"}
          </div>
          <p className="mb-3 mt-3 text-sm text-muted">
            The model scores this URL {(r.probability * 100).toFixed(1)}% on a scale where higher means more like phishing.
            Anything at or above {(r.threshold * 100).toFixed(0)}% is flagged. It is a statistical guess from the URL text
            alone, not a certainty or a measured chance; the site was never visited.
          </p>
          <dl className="m-0 grid grid-cols-[5.5rem_1fr] gap-x-3 gap-y-1 font-mono text-xs">
            {(["scheme", "host", "path", "query", "fragment"] as const).map((k) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd className="m-0 break-all">{parts[k] || <span className="text-muted">—</span>}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}

function Explanation({ r }: { r: Prediction }) {
  return (
    <section className="rise rounded-lg border border-line bg-panel p-5">
      <h2 className="m-0 text-lg font-semibold">How this prediction was made</h2>
      <p className="mb-4 mt-1 max-w-3xl text-sm text-muted">
        The URL is turned into 30 numeric features. Each one is compared with what a typical legitimate URL looks like:
        we test how the phishing score changes when that feature alone is swapped for the typical value, and when it
        alone is added to a typical URL. Red bars pushed the score up; green bars pulled it down. Effects interact,
        so bars are an estimate and do not add up exactly.
      </p>
      <div className="grid gap-8 lg:grid-cols-[3fr_2fr]">
        <div>
          <h3 className="mb-3 mt-0 text-xs font-semibold uppercase tracking-wider text-muted">Top features</h3>
          <ContributionChart items={r.explanation.top_features} />
        </div>
        <div>
          <h3 className="mb-3 mt-0 text-xs font-semibold uppercase tracking-wider text-muted">By feature group</h3>
          <GroupChart groups={r.explanation.group_contributions} />
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    getModel().then(setModel).catch(() => {});
    getMetrics().then(setMetrics).catch(() => {});
  }, []);

  async function run(target: string) {
    const value = target.trim();
    if (!value) return setError("Enter a URL to check.");
    if (value.length > MAX_URL_LENGTH) return setError(`URLs longer than ${MAX_URL_LENGTH} characters are not supported.`);
    setBusy(true);
    setError(null);
    try {
      setResult(await predict(value));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6">
      <header className="mb-8">
        <p className="m-0 font-mono text-xs uppercase tracking-[0.25em] text-accent">phishing-detector</p>
        <h1 className="mb-2 mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Is this URL safe?</h1>
        <p className="m-0 max-w-2xl text-muted">
          Paste a link. A machine-learning model scores it from the characters in the URL alone, and shows which
          signals drove the verdict. Nothing is fetched or opened.
        </p>
      </header>

      <form onSubmit={(e) => { e.preventDefault(); run(url); }} className="mb-3 flex flex-col gap-2 sm:flex-row">
        <label htmlFor="url" className="sr-only">URL to check</label>
        <input
          id="url" value={url} onChange={(e) => setUrl(e.target.value)} spellCheck={false} autoComplete="off"
          placeholder="https://example.com/login"
          aria-invalid={!!error} aria-describedby={error ? "url-error" : undefined}
          className="min-w-0 flex-1 rounded-lg border border-line bg-panel px-4 py-3 font-mono text-sm outline-none placeholder:text-muted/60 focus:border-accent focus:ring-2 focus:ring-accent/30"
        />
        <button type="submit" disabled={busy}
          className="rounded-lg bg-accent px-6 py-3 font-semibold text-slate-950 transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-60">
          {busy ? "Analyzing…" : "Analyze"}
        </button>
      </form>

      <div className="mb-8 flex flex-wrap items-center gap-2 text-xs text-muted">
        <span>Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" onClick={() => { setUrl(ex); run(ex); }}
            className="max-w-[16rem] truncate rounded border border-line px-2 py-1 font-mono hover:border-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-accent"
            title={ex}>
            {ex}
          </button>
        ))}
      </div>

      {error && (
        <p id="url-error" role="alert" className="mb-6 rounded-lg border border-danger/50 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}
        </p>
      )}

      {result && (
        <div className="mb-12 space-y-4">
          <Verdict r={result} />
          <Explanation r={result} />
        </div>
      )}

      {model && <ModelPanel model={model} metrics={metrics} />}
    </main>
  );
}
