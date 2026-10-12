export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const MAX_URL_LENGTH = 4096;

export type Contribution = {
  feature: string;
  group: string;
  value: number;
  legit_median: number;
  contribution: number;
};

export type Prediction = {
  label: "phishing" | "legitimate";
  probability: number;
  threshold: number;
  model_version: string;
  algorithm: string;
  features: Record<string, number>;
  explanation: { top_features: Contribution[]; group_contributions: Record<string, number> };
  url_parts: { scheme: string; host: string; path: string; query: string; fragment: string };
};

export type Confusion = { tn: number; fp: number; fn: number; tp: number };

export type Threshold = { threshold: number; fnr: number; recall: number; precision: number; f1: number; rule?: string };

export type ModelInfo = {
  version: string;
  algorithm: string;
  trained_at: string;
  feature_schema_version: number;
  train_rows: number;
  hyperparameters: Record<string, string | number | null>;
  threshold: Threshold;
  group_importance: Record<string, number>;
  feature_importance: Record<string, number>;
  candidates: { algo: string; cv_f1: number; cv_ap: number; cv_roc_auc: number }[];
  test_metrics: ({ confusion_matrix: Confusion } & Record<string, number | Confusion>) | null;
};

export type SweepRow = { threshold: number; precision: number; recall: number; f1: number; fnr: number };

export type TestMetrics = {
  test_rows: number;
  selected: string;
  selected_at_chosen_threshold: Record<string, number> & { confusion_matrix: Confusion };
};

export type ComparisonRow = {
  model: string; threshold: number; accuracy: number; precision: number; recall: number; f1: number;
  specificity: number; fnr: number; roc_auc: number; pr_auc: number;
};

export type AblationRow = {
  setting: string; kind: "baseline" | "drop_one" | "cumulative"; removed: string; n_features: number;
  roc_auc: number; pr_auc: number; f1: number; recall: number; d_pr_auc: number; d_f1: number;
};

export type CurveSet = { name: string; roc_auc: number; pr_auc: number; roc: [number, number][]; pr: [number, number][] };
export type ReliabilityRow = { bin: string; n: number; mean_pred: number; actual_phishing: number; model: "raw" | "final" };
export type Calibration = { ece: number; brier: number; log_loss: number };

export type ExternalCell = { n: number; recall?: number; false_positive_rate?: number };
export type ExternalRun = { version: string; threshold: number } & Record<string, ExternalCell | string | number>;

export type Metrics = {
  test: TestMetrics | null;
  threshold_sweep: SweepRow[];
  comparison: ComparisonRow[];
  ablation: AblationRow[];
  calibration: { summary: { raw: Calibration; final: Calibration; method: string } | null; reliability: ReliabilityRow[] };
  curves: Record<string, CurveSet> | null;
  external: ExternalRun | null;
};

export type HistoryItem = {
  id: number; url_hash: string; label: "phishing" | "legitimate"; probability: number;
  model_version: string; created_at: string;
};
export type History = { total: number; limit: number; offset: number; items: HistoryItem[] };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error("Cannot reach the detection service. Is the backend running?");
  }
  if (res.status === 422) throw new Error("That does not look like a valid URL (empty or over 4096 characters).");
  if (res.status === 503) throw new Error("The model is not available yet. Training may still be running.");
  if (!res.ok) throw new Error(`Request failed (${res.status}).`);
  return res.json() as Promise<T>;
}

export const predict = (url: string) =>
  request<Prediction>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

export const getModel = () => request<ModelInfo>("/models");
export const getMetrics = () => request<Metrics>("/metrics");
export const getHistory = (limit: number, offset: number) =>
  request<History>(`/predictions?limit=${limit}&offset=${offset}`);
