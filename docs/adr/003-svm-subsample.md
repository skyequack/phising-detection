# ADR 003: Train the SVM on a 50,000-row subsample; search the Random Forest less

Status: accepted (2026-10-07), consequences re-measured on the v2 data (2026-10-11)

## Context
Five algorithms are compared with 5-fold cross-validated random search. An RBF-kernel SVM scales between O(n²) and O(n³)
in the number of rows; the training set has 350,141 rows (466,429 in v2). Tuning it on all rows was not feasible on one
machine in a reasonable time. A Random Forest fit on the full data also takes minutes per candidate.

## Decision
- SVM: tune and fit on a stratified 50,000-row subsample (fixed seed), 8 candidates.
- Random Forest: 4 search candidates instead of 8 for the others.
- Everything else uses the full training split.

## Consequences
- The comparison is not like for like. The SVM saw 11% (v2) of the data the other models saw, so its scores understate what
  an SVM could do. In the v2 run its CV F1 was 0.792 against 0.858 for the Random Forest, but that gap mixes the algorithm
  and the data it was given.
- The Random Forest was searched less than the other tree models, so its tuning may be less thorough. It was still selected
  and its depth and size are large (300 trees, depth 30), which is why the model file is 1.1 GB.
- The selected model does not depend on the SVM, so the final results are unaffected; only the algorithm comparison table
  carries the caveat. It is disclosed in the report notes ([PROGRESS.md](../PROGRESS.md)).
- SVM predictions use a Platt-scaled probability, fitted by an internal cross-validation, which adds fit time.

## Alternatives considered
Linear SVM on all rows (different model class from the one proposed); approximate kernels such as Nystroem (adds another
tuned component); dropping the SVM (the report asks for it). A larger subsample would narrow the gap at a quadratic cost.
