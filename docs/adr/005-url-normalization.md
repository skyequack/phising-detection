# ADR 005: Normalize scheme, `www.` and a bare trailing slash before extracting features

Status: accepted (2026-10-07)

## Context
The first models scored F1 of 0.97 to 0.99, then flagged ordinary sites such as `https://github.com` as phishing.
Measured on the training split of the original dataset:

| Signal | Legitimate | Phishing |
|---|---|---|
| Scheme is `https` | 100.0% | 6.4% |
| Host starts with `www.` | 99.8% | 8.1% |
| Bare two-label host | 0.1% | 57.1% |

These come from how the dataset was collected, not from how phishing works. A model can score highly by learning "not
`https://www.` means phishing", and that does not carry over to real URLs.

## Decision
`ml.features.normalize_url` runs before any feature is computed: the scheme becomes `http://`, a leading `www.` and a bare
trailing slash after the host are removed, and other schemes are left untouched. The same function runs at training
and at prediction time, and the dataset cleaning uses a similar canonical key to deduplicate before splitting
([dataset README](../dataset/README.md)). Feature schema version 2 ([feature-schema.md](../design/feature-schema.md)).

## Consequences
- Held-out F1 fell from about 0.97 to about 0.83 on the original data. The lower figure is the honest one.
- The model can no longer use `https` versus `http` at all, a real signal in practice that is removed here only because
  the dataset makes it misleading. Any report must say this.
- `https://x.com/a` and `http://www.x.com/a` produce the same features and the same score, and the same stored hash, which a
  test checks.
- Normalization removes the shortcuts it names, not all of them. The legitimate class still came from one narrow source;
  on real links the model flagged 18.8% before the v2 retrain and 14.8% after ([PROGRESS.md](../PROGRESS.md)).
- Any later change to normalization changes the feature schema and invalidates saved models; the schema version is stored
  in the model metadata and checked at load time.

## Alternatives considered
Dropping the scheme and `www.` features only (leaves the bare-host and slash signals); rebalancing or resampling the data
(does not remove a systematic collection bias); collecting a second legitimate source first (done in v2 as an addition,
not a replacement).
