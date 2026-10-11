import Link from "next/link";
import type { Metrics, ModelInfo } from "@/lib/api";
import { Card, ConfusionMatrix, Importance, ThresholdSweep } from "./charts";

export default function ModelPanel({ model, metrics }: { model: ModelInfo; metrics: Metrics | null }) {
  const cm = model.test_metrics?.confusion_matrix;
  const sweep = metrics?.threshold_sweep ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="m-0 text-lg font-semibold">The model</h2>
        <span className="font-mono text-sm text-muted">
          {model.algorithm} · {model.version} · threshold {model.threshold.threshold.toFixed(2)}
        </span>
        <Link href="/model" className="text-sm text-accent hover:underline">Model details</Link>
        <Link href="/metrics" className="text-sm text-accent hover:underline">All metrics</Link>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Card title="What the model relies on" note="Share of overall importance by feature group.">
          <Importance groups={model.group_importance} />
        </Card>
        {sweep.length > 0 && (
          <Card title="Choosing the threshold" note="Measured on out-of-fold training predictions, never the test set.">
            <ThresholdSweep rows={sweep} chosen={model.threshold.threshold} />
          </Card>
        )}
        {cm && (
          <Card title="Held-out test results" note="Counts on the unseen 20% test split at the chosen threshold.">
            <ConfusionMatrix c={cm} />
          </Card>
        )}
      </div>
    </div>
  );
}
