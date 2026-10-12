"use client";

import { Card, Importance, Notice, PageHeader, Stat, pct } from "@/components/charts";
import { pretty } from "@/components/ContributionChart";
import { getModel, type ModelInfo } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";

const ALGO_NAMES: Record<string, string> = {
  lr: "Logistic Regression", dt: "Decision Tree", rf: "Random Forest", svm: "SVM (RBF)", xgb: "XGBoost",
};

function Candidates({ model }: { model: ModelInfo }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[26rem] border-collapse text-left text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wider text-muted">
            <th scope="col" className="py-2 pr-3 font-semibold">Algorithm</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">CV F1</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">CV PR-AUC</th>
            <th scope="col" className="py-2 pl-3 text-right font-semibold">CV ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {model.candidates.map((c) => {
            const name = ALGO_NAMES[c.algo] ?? c.algo;
            const picked = name === model.algorithm;
            return (
              <tr key={c.algo} className={`border-t border-line ${picked ? "bg-accent/10" : ""}`}>
                <th scope="row" className="py-2 pr-3 font-medium">
                  {name}
                  {picked && <span className="ml-2 text-xs text-accent">selected</span>}
                </th>
                <td className="px-3 py-2 text-right font-mono">{c.cv_f1.toFixed(3)}</td>
                <td className="px-3 py-2 text-right font-mono">{c.cv_ap.toFixed(3)}</td>
                <td className="py-2 pl-3 text-right font-mono">{c.cv_roc_auc.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TopFeatures({ importance }: { importance: Record<string, number> }) {
  const top = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const max = Math.max(...top.map((t) => t[1]), 1e-9);
  return (
    <ul className="m-0 list-none space-y-2 p-0">
      {top.map(([f, v]) => (
        <li key={f} className="grid grid-cols-[minmax(0,10rem)_1fr_3rem] items-center gap-3 text-sm">
          <span className="truncate font-mono text-[13px]" title={f}>{pretty(f)}</span>
          <div className="h-3 rounded bg-line/40">
            <div className="h-full rounded bg-accent" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono text-xs text-muted">{pct(v, 1)}</span>
        </li>
      ))}
    </ul>
  );
}

export default function ModelPage() {
  const { data: m, error } = useFetch(getModel);
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6">
      <PageHeader kicker="model" title="The model behind the verdicts">
        What was trained, how it was chosen, and what it relies on. Measured results are on the Metrics page.
      </PageHeader>
      {error && <Notice kind="error">{error}</Notice>}
      {!error && !m && <Notice>Loading…</Notice>}
      {m && (
        <div className="space-y-4">
          <section className="rounded-lg border border-line bg-panel p-5">
            <h2 className="m-0 text-lg font-semibold">{m.algorithm}</h2>
            <p className="mb-4 mt-1 font-mono text-sm text-muted">{m.version}</p>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="Trained" value={new Date(m.trained_at).toLocaleDateString()} />
              <Stat label="Training URLs" value={m.train_rows.toLocaleString()} />
              <Stat label="Features" value={String(Object.keys(m.feature_importance).length)} note={`feature schema ${m.feature_schema_version}`} />
              <Stat label="Decision threshold" value={m.threshold.threshold.toFixed(2)} note="a score at or above this is flagged" />
            </div>
            <p className="mb-0 mt-4 max-w-3xl text-sm text-muted">
              On the training data (out-of-fold predictions) this threshold gave precision {pct(m.threshold.precision)} and
              recall {pct(m.threshold.recall)}. It was picked by maximising F1 on those predictions, never on the test set.
              Results on unseen data are lower; see the Metrics page.
            </p>
          </section>

          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Feature groups" note="Share of the model's overall importance.">
              <Importance groups={m.group_importance} />
            </Card>
            <Card title="Most important features" note="Share of overall importance, top 12 of 30.">
              <TopFeatures importance={m.feature_importance} />
            </Card>
            <Card title="How it was chosen"
              note="Five algorithms, tuned with 5-fold cross-validation on the training split. Best F1 wins; near-ties go to the more interpretable model.">
              <Candidates model={m} />
            </Card>
            <Card title="Settings" note="Hyperparameters picked by random search.">
              <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs">
                {Object.entries(m.hyperparameters).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-muted">{k.replace(/^clf__/, "")}</dt>
                    <dd className="m-0">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          </div>
          <p className="text-xs text-muted">
            The model sees only the characters of the URL, never the page. The explanation on the Detector page is an
            estimate made by swapping each feature for a typical value, not a reading of the model&apos;s internals.
          </p>
        </div>
      )}
    </main>
  );
}
