# ADR 004: Explain predictions with a model-agnostic swap heuristic

Status: accepted (2026-10-07)

## Context
The dashboard shows which features pushed a URL toward phishing or legitimate. The explanation has to work for any of the
five candidate algorithms, run inside a request, and be understandable without machine-learning background. The saved
model's own importances are global (the same for every URL), so they cannot explain one verdict.

## Decision
For each of the 30 features, compare the URL with the median of legitimate training URLs (`legit_medians` in
`metadata.json`) in two ways, using the calibrated model:
1. **Reset:** the drop in the score when only that feature is replaced by the legitimate median.
2. **Add-in:** the rise when only that feature takes its real value on an otherwise all-median row.

The contribution is the average of the two. All 62 rows (the URL, the all-median baseline, and 30 resets and 30 add-ins) are scored in one batch (`backend/service.py`). The top 8 features
and the sum per feature group are returned.

## Consequences
- Works for every algorithm and needs no extra dependency.
- Using both directions keeps the bars informative when the model is saturated (a single reset of a clearly phishing URL
  often changes nothing). Averaging them is a design choice made here; it is not a published method.
- **The bars are estimates.** Features interact, and the baseline is a synthetic all-median row, so contributions do not
  sum to the score. The UI says so.
- **They explain this model, not the world.** A feature with a large bar is one the model relies on, which may reflect a
  quirk of the training data. The ablation on the Metrics page shows how much each group matters overall.
- Cost: one request scores 62 rows, which is part of the roughly 26 to 30 ms per prediction. The forest is limited to one
  thread per request at serving time to keep that low.
- The explanation is computed from the unthresholded score, so a URL just under the threshold can show bars that look as
  strong as one just over it.

## Alternatives considered
SHAP (exact for trees, but a new dependency, and its cost on 300 deep trees was not evaluated); LIME (random sampling, so
the same URL can explain differently on each call); built-in importances (global only). SHAP is the obvious upgrade if
exact attributions matter.
