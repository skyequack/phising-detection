"use client";

import {
  Card, ConfusionMatrix, DeltaBars, LineChart, Notice, PALETTE, PageHeader, Stat, ThresholdSweep, pct,
  type Series,
} from "@/components/charts";
import { getMetrics, type ExternalCell, type ExternalRun, type Metrics } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";

const EXTERNAL_LABELS: Record<string, string> = {
  phishtank_newest_15pct: "Newest PhishTank phishing (Jul–Oct 2026)",
  openphish_feed: "OpenPhish live feed",
  crawl_links: "Real links crawled from popular sites",
  "sanity_legit": "Hand-picked legitimate URLs",
  "sanity_phishing": "Hand-made phishing URLs",
  "tranco_1-1000": "Popular domains, ranks 1–1k (bare)",
  "tranco_1001-10000": "Popular domains, ranks 1k–10k (bare)",
  "tranco_10001-100000": "Popular domains, ranks 10k–100k (bare)",
  "tranco_100001-1000000": "Popular domains, ranks 100k–1M (bare)",
};

const cell = (run: ExternalRun, key: string) => run[key] as ExternalCell;

function External({ ext }: { ext: ExternalRun }) {
  const keys = Object.keys(ext).filter((k) => typeof ext[k] === "object");
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-0 border-collapse text-left text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wider text-muted">
            <th scope="col" className="py-2 pr-3 font-semibold">Data the model has never seen</th>
            <th scope="col" className="hidden px-3 py-2 text-right font-semibold sm:table-cell">URLs</th>
            <th scope="col" className="hidden px-3 py-2 font-semibold sm:table-cell">Measures</th>
            <th scope="col" className="py-2 pl-2 text-right font-semibold sm:pl-3">Result</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => {
            const c = cell(ext, k);
            const caught = c.recall !== undefined;
            const value = caught ? c.recall : c.false_positive_rate;
            return (
              <tr key={k} className="border-t border-line">
                <th scope="row" className="py-2 pr-3 font-normal">
                  {EXTERNAL_LABELS[k] ?? k}
                  <span className="block text-xs text-muted sm:hidden">
                    {c.n.toLocaleString()} URLs · {caught ? "phishing caught ↑" : "wrongly flagged ↓"}
                  </span>
                </th>
                <td className="hidden px-3 py-2 text-right font-mono sm:table-cell">{c.n.toLocaleString()}</td>
                <td className="hidden px-3 py-2 text-xs text-muted sm:table-cell">{caught ? "phishing caught ↑" : "legitimate wrongly flagged ↓"}</td>
                <td className="py-2 pl-2 text-right font-mono sm:pl-3">{value === undefined ? "—" : pct(value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Comparison({ rows }: { rows: Metrics["comparison"] }) {
  const cols = [["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"]] as const;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] border-collapse text-left text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wider text-muted">
            <th scope="col" className="py-2 pr-3 font-semibold">Model</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Threshold</th>
            {cols.map(([, h]) => <th key={h} scope="col" className="px-3 py-2 text-right font-semibold">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const tuned = r.model.includes("@");
            return (
              <tr key={r.model} className={`border-t border-line ${tuned ? "bg-accent/10" : ""}`}>
                <th scope="row" className="py-2 pr-3 font-medium">{r.model}</th>
                <td className="px-3 py-2 text-right font-mono">{r.threshold.toFixed(2)}</td>
                {cols.map(([k]) => <td key={k} className="px-3 py-2 text-right font-mono">{r[k].toFixed(3)}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function MetricsPage() {
  const { data: m, error } = useFetch(getMetrics);
  const t = m?.test?.selected_at_chosen_threshold;
  const curves = m?.curves ? Object.entries(m.curves) : [];
  const roc: Series[] = curves.map(([, c], i) => ({ name: `${c.name} (${c.roc_auc.toFixed(3)})`, color: PALETTE[i % PALETTE.length], points: c.roc }));
  const pr: Series[] = curves.map(([, c], i) => ({ name: `${c.name} (${c.pr_auc.toFixed(3)})`, color: PALETTE[i % PALETTE.length], points: c.pr }));
  const rel = m?.calibration.reliability ?? [];
  const calib: Series[] = (["raw", "final"] as const).map((k, i) => ({
    name: k === "raw" ? "Before calibration" : "After calibration", color: PALETTE[i + 1], dots: true,
    points: rel.filter((r) => r.model === k).map((r): [number, number] => [r.mean_pred, r.actual_phishing]),
  }));
  const dropOne = (m?.ablation ?? []).filter((r) => r.kind === "drop_one");
  const cumulative = (m?.ablation ?? []).filter((r) => r.kind === "cumulative");

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6">
      <PageHeader kicker="metrics" title="How well does it work?">
        Measured results for the model in use. The first section matters most: scores on the held-out test further down are
        higher than what the model achieves on URLs from other sources.
      </PageHeader>
      {error && <Notice kind="error">{error}</Notice>}
      {!error && !m && <Notice>Loading…</Notice>}
      {m && (
        <div className="space-y-4">
          {m.external && (
            <Card title="On data it has not seen"
              note="Same model, URLs from sources it was never trained or tuned on. This is the more realistic picture.">
              <External ext={m.external} />
              <p className="mb-0 mt-3 max-w-3xl text-xs text-muted">
                Higher is better for phishing caught, lower for legitimate wrongly flagged (the arrows show the direction that is better). The popular-domain rows are bare
                domains, and the model was trained on similar bare domains from other ranks, so they flatter it. The crawled
                links, which include real paths, are the fairer measure of false alarms. Samples are small, so treat
                the percentages as rough.
              </p>
            </Card>
          )}

          {t && (
            <Card title="Held-out test results"
              note={`${m.test?.test_rows.toLocaleString()} URLs set aside before training, from the same source as the training data. Scores here are higher than on other sources.`}>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                <Stat label="Precision" value={pct(t.precision)} note="flagged that really were phishing" />
                <Stat label="Recall" value={pct(t.recall)} note="phishing that was flagged" />
                <Stat label="F1" value={t.f1.toFixed(3)} />
                <Stat label="ROC-AUC" value={t.roc_auc.toFixed(3)} />
                <Stat label="PR-AUC" value={t.pr_auc.toFixed(3)} />
              </div>
              <div className="mt-4 grid gap-6 md:grid-cols-2">
                <div>
                  <h3 className="mb-2 mt-0 text-xs font-semibold uppercase tracking-wider text-muted">Confusion matrix</h3>
                  <ConfusionMatrix c={t.confusion_matrix} />
                </div>
                {m.threshold_sweep.length > 0 && t && (
                  <div>
                    <h3 className="mb-2 mt-0 text-xs font-semibold uppercase tracking-wider text-muted">Choosing the threshold</h3>
                    <ThresholdSweep rows={m.threshold_sweep} chosen={t.threshold} />
                  </div>
                )}
              </div>
            </Card>
          )}

          {m.comparison.length > 0 && (
            <Card title="Algorithm comparison"
              note="All five on the same held-out test. The first five use a plain 0.5 cut-off and uncalibrated scores; the highlighted row is the model in use, with its chosen threshold and calibration.">
              <Comparison rows={m.comparison} />
            </Card>
          )}

          {curves.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card title="ROC curves" note="Closer to the top-left is better; the dashed line is guessing. AUC in brackets.">
                <LineChart series={roc} diagonal xLabel="False positive rate" yLabel="True positive rate"
                  label="ROC curves for the five algorithms on the test set" />
              </Card>
              <Card title="Precision–recall curves" note="Closer to the top-right is better. This view matters more when phishing is the rarer class.">
                <LineChart series={pr} xLabel="Recall" yLabel="Precision" label="Precision-recall curves for the five algorithms on the test set" />
              </Card>
            </div>
          )}

          {rel.length > 0 && m.calibration.summary && (
            <Card title="Are the percentages trustworthy?"
              note="Of the URLs given about X% by the model, how many were really phishing? On the diagonal means the percentage can be read at face value.">
              <div className="grid gap-6 md:grid-cols-[minmax(0,22rem)_1fr]">
                <LineChart series={calib} diagonal xLabel="Model score" yLabel="Share actually phishing"
                  label="Reliability curve before and after calibration" />
                <div className="space-y-3 text-sm text-muted">
                  <p className="m-0">
                    Calibration error fell from {m.calibration.summary.raw.ece.toFixed(3)} to {m.calibration.summary.final.ece.toFixed(3)}
                    {" "}({m.calibration.summary.method} correction). That holds for URLs like the training data. On URLs from
                    elsewhere the scores are probably too confident, so read them as a ranking, not a probability.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {dropOne.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card title="What happens without a feature group"
                note="Change in PR-AUC when one group is removed and the model is retrained. Longer red bars mean the group matters more.">
                <DeltaBars rows={dropOne.map((r) => ({ label: r.removed.replace(/_/g, " "), value: r.d_pr_auc, hint: r.setting }))} />
              </Card>
              <Card title="Removing the most important groups in turn"
                note="Cumulative: each row removes one more group, most important first.">
                <DeltaBars rows={cumulative.map((r) => ({ label: r.setting, value: r.d_pr_auc, hint: `removes ${r.removed}` }))} />
              </Card>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
