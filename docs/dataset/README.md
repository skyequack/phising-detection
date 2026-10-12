# Dataset: URL dataset.csv

Location: `data/raw/URL dataset.csv` (raw, never edited in place; cleaned output goes to `data/processed/`).

Fills in the Table 3.1 items from the report. Figures below were measured on 2026-10-07 from the raw file.

## Schema

| Column | Type | Description |
|---|---|---|
| `url` | string | Full URL, including scheme. |
| `type` | string (categorical) | Class label: `legitimate` or `phishing`. |

- Format: UTF-8 CSV with a header row, 2 columns, no index column.
- 450,176 rows, 32,312,600 bytes.
- The dataset contains **no pre-computed features**, only the URL string and the label. All ~30 features must come from `ml/features.py`, which also removes the training/inference mismatch risk the report describes in Section 2.4.
- SHA-256: `accb2dfbfd3329a8b5cb1b85dcad90314a660c272be755cb45d0f6865014b466`

## Profile (raw)

| Item | Value |
|---|---|
| Rows | 450,176 |
| `legitimate` | 345,738 (76.8%) |
| `phishing` | 104,438 (23.2%) |
| Empty / missing URLs | 0 |
| Malformed rows (column count ≠ 2) | 0 |
| Exact duplicate URLs | 0 |
| URLs with conflicting labels | 0 |
| URL length (chars) | min 8, max 2,314 (none above the 4,096 input limit) |
| Schemes | `https` 352,185, `http` 97,948, `httpss` 35, `ftp` 8 |

## Validation findings (2026-10-07)

File integrity is clean: valid UTF-8, no BOM or NUL bytes, LF line endings, every row has exactly 2 columns, header present, no empty URL or label.

| Check | Count | Action |
|---|---|---|
| Canonical duplicates (ignoring scheme, `www.`, trailing `/`, case) | 12,447 | Dedupe after canonicalizing, before the split. |
| Canonical label conflicts | 3 | Drop these pairs (see below). |
| Case-insensitive duplicates (exact-match dedupe misses them) | 81, no label conflicts | Covered by canonical dedupe. |
| URLs containing whitespace | 521, all phishing | Keep or drop deliberately. A whitespace-containing URL is itself a phishing signal, but it can leak the label. |
| Control characters | 2 (`\x13`) | Strip or drop. |
| Non-ASCII characters | 310 | Keep. Relevant to punycode/homograph features. |
| `urlsplit` raises `ValueError` | 1: `http://ladiesfirst-privileges[.]com/...` (invalid IPv6 bracket) | The extractor must catch parse errors and fall back, not crash. |
| No hostname | 3: `http://#NAME?`, `http://?`, `http://?guid=Windows Updates Manager` (all phishing) | Drop. These are spreadsheet/scraper artefacts. |
| Injected markup | 1 (`...php""></script><script>`) | Keep as test input for the XSS-safety tests. |
| IP-address hosts | 3,954 | Valid signal for `has_ip_host`. |
| Explicit port / `@` in URL | 1,606 / 2,955 | Valid signals for `has_port`, `has_at_symbol`. |
| `httpss://` (35, all legitimate) and `ftp://` (8) | 43 | Drop or repair. All 35 `httpss` rows are labelled legitimate, so a model could learn "malformed scheme means legitimate". |

The 3 canonical label conflicts:
- `https://www.4shared.com` (legitimate) vs `http://4shared.com` (phishing)
- `https://www.tommyhumphreys.com/` (legitimate) vs `http://tommyhumphreys.com/` (phishing)
- `https://www.google.com/webhp` (legitimate) vs `http://www.google.com/webhp` (phishing)

## Cleaning results (`python -m ml.data_prep`, seed 42)

Implemented in `ml/data_prep.py`, tested in `tests/test_data_prep.py`. Output in `data/processed/` (git-ignored): `train.parquet`, `test.parquet` (columns `url`, `label`), `dropped.csv` (every removed row with its reason), `cleaning_report.json`. Re-running gives byte-identical frames.

| Drop reason | Rows | Notes |
|---|---|---|
| `bad_scheme` | 43 | `httpss://` (35) and `ftp://` (8). 36 legitimate, 7 phishing. |
| `control_chars` | 2 | Both phishing. |
| `no_hostname` | 3 | Both phishing. |
| `unparseable` | 1 | `[.]` URL, phishing. |
| `label_conflict` | 6 | 3 groups, whole group removed (3 legitimate, 3 phishing). |
| `duplicate` | 12,444 | Canonical duplicates, first occurrence kept. 127 legitimate, **12,317 phishing**. |
| **Total dropped** | **12,499** | |

| | Legitimate | Phishing | Phishing share |
|---|---|---|---|
| Raw | 345,738 | 104,438 | 23.2% |
| Clean | 345,572 | 92,105 | 21.0% |
| Train (80%) | 276,457 | 73,684 | 21.0% |
| Test (20%) | 69,115 | 18,421 | 21.0% |

- **Dedupe removed almost only phishing rows** (11.8% of phishing vs 0.05% of legitimate). The phishing set contained many scheme, `www.` and trailing-slash variants of the same URL. The imbalance is now about 3.75:1, not 3.3:1.
- The canonical key ignores scheme, so `http://x.com` and `https://x.com` count as the same URL. If a site is phishing on `http` and legitimate on `https`, both rows are dropped as a conflict.
- Dedupe ran before the split, so no canonical URL appears in both train and test (checked: overlap 0).
- Whitespace, IP-host, port, `@` and non-ASCII URLs are kept as real signals.

## Shortcut signals (found after the first model, 2026-10-07)

The first models scored 97-99% F1, but they flagged ordinary sites such as `https://github.com` as phishing. Measured on the training split:

| Signal | Legitimate | Phishing |
|---|---|---|
| Scheme is `https` | 100.0% | 6.4% |
| Host starts with `www.` | 99.8% | 8.1% |
| Bare two-label host (`example.com`) | 0.1% | 57.1% |
| URL is `scheme://host/` only (lone trailing slash) | 13.0% | 5.4% |

A model can score highly on this dataset by learning "not `https://www.` means phishing". Those scores do not carry over to real URLs. Fix: `ml/features.py` normalizes every URL before extracting features (scheme forced to `http://`, leading `www.` and a lone trailing `/` removed), see `docs/design/feature-schema.md`. After the fix the held-out F1 fell from about 0.97 to 0.83. The model cannot use the http/https distinction at all, which is a real loss in practice but a pure artefact here.

Remaining concern: the legitimate class comes from one collection source and is narrow. On 40 hand-made real legitimate URLs (`tests/data/sanity_urls.json`) the final model still flags 10 (25%). Treat all reported metrics as describing this dataset only, and state that in the report. This was addressed in v2 with Tranco and PhishTank, see "Additional sources (v2)" below; on 921 real crawled links the v2 model still flags 14.8%.

## Notes for preprocessing

- **Class imbalance (~3.3:1 raw, ~3.75:1 after cleaning).** Stratified splits and stratified 5-fold CV are used; imbalance is handled with class weights. Report PR-AUC, recall, and FNR, not accuracy alone (the report's Section 3.7 already requires this).
- **Label normalization.** Labels are lowercase strings. Map to `legitimate`=0 and `phishing`=1 (positive class = phishing).
- **Canonicalization may expose duplicates.** There are no exact duplicates, but canonicalizing (case, trailing slash, scheme, `www.`) can create near-duplicates and conflicting labels. Re-check after canonicalizing, before the split (report Section 3.4).
- **Malformed schemes.** 35 `httpss://` URLs. Decide whether to drop or repair them. `urllib.parse` will parse them, but they are likely noise.
- **Collection-bias check.** Measured: 85.0% of legitimate and 91.6% of phishing URLs have a path, and 13.3% / 18.7% have a query, so structure alone doesn't trivially separate the classes. The two classes may still differ systematically (for example, the top of the file looks like popular-site bare hosts), so check the path and query feature groups in the ablation study and mention it as a limitation. See "Shortcut signals" above: the collection bias turned out to be much larger than this check suggested.
- **Order.** The file looks sorted by label (legitimate first). Shuffle with a fixed seed before any non-stratified operation.

## Table 3.1 status

| Item | Status |
|---|---|
| Source | "Phishing URL dataset", Version 1, published 2 April 2024. Contributors: Jishnu K S Kaitholikkal and Arthi B. DOI: [10.17632/vfszbj9b36.1](https://doi.org/10.17632/vfszbj9b36.1) (Mendeley Data DOI prefix). Source details were supplied by the user. |
| License | CC BY 4.0 (supplied by the user). Attribution to the contributors above is required when the dataset is used or redistributed. |
| Sample count | 450,176 raw, 437,677 after cleaning (train 350,141 / test 87,536). See "Cleaning results". |
| Class distribution | 345,738 legitimate / 104,438 phishing (raw). |
| Label definition | As defined by the original publishers. The labelling method is not in the supplied metadata, so check the DOI page's description. |
| Duplicate policy | Exact duplicates removed, conflicting labels flagged (none in the raw file). |
| Train/test split | 80/20, stratified by label, fixed random seed. |

The report names the UCI Phishing Websites and PhiUSIIL datasets, but this file is neither (it has no feature columns). The report text in Section 3.4 and the references should be updated to cite the dataset actually used (10.17632/vfszbj9b36.1).

## Additional sources (v2, added 2026-10-08)

Accessed 2026-10-08, stored git-ignored under `data/external/` and never edited. Used only as described; none is redistributed in this repository.

| Source | File | SHA-256 | Size / content | Used for |
|---|---|---|---|---|
| PhishTank `online-valid.csv.gz` | `data/external/phishtank/` | `bb3b486e1cb4052aa1e8635ae6fbf3f7d4a6c2fb1b2a277ac00f0ec00c62b50f` | 2.4 MB; 69,623 verified online phishing URLs, submitted 2011 to 2026-10-07 | Phishing training (older 85% by submission time, 57,134 rows after cleaning) and held-out test (newest 15%, 10,083 rows) |
| Tranco top-1M list (`top-1m.csv.zip`) | `data/external/tranco/` | `822131033446fe75743fcd96d044df2751901465f090c422ef118e04c3f68a0c` | 9.6 MB; ranked domains, no paths | Legitimate training (59,156 domains with template paths), validation domains (2,000, bare), and the link-crawl domains |
| OpenPhish community feed (`feed.txt`) | `data/external/openphish/` | `c00074e37dce95c9eb44ca64d033b82b5373154f9e4078f8f6eb39e9cb2e47a7` | 14 KB; 300 URLs | External phishing test only, never trained on |
| Link crawl (`ml/crawl_links.py`) | `data/external/crawl/` | n/a | 921 same-site links from the front pages of 191 of 500 sampled Tranco domains (robots.txt honoured, one page per domain) | External legitimate test only |

Derived files are in `data/processed/` (`phishtank_train.parquet`, `phishtank_heldout.parquet`, `phishtank_report.json`) and `data/training/` (`train.parquet`, `test.parquet`, `build_report.json`). Scripts: `ml/phishtank_prep.py`, `ml/crawl_links.py`, `ml/build_training_set.py`. The PhishTank cleaning applies the same rules as the original data, plus HTML-unescaping and a 4,096-character limit; 2,403 of 69,623 rows were dropped (2,370 duplicates, 31 already in the original dataset, 1 in OpenPhish, 1 control characters, 2 too long, with `too_long` now part of `ml.data_prep`).

Caveats:
- The Tranco paths are template-generated (`ml/build_training_set.py`), not real URLs. The crawl is the check on whether they generalise.
- The Tranco file used is the default list downloaded on 2026-10-08; no Tranco list ID or permalink was recorded, so the exact list cannot be re-fetched. Use the SHA-256 above to identify it.
- PhishTank was downloaded without an application key (the site asks automated downloads to register one and limits keyless downloads). One download was made.

### Licences and terms (checked 2026-10-08 from the sites' public pages; not legal advice)

| Source | What the public pages say | Consequence for this project |
|---|---|---|
| PhishTank | The data page and home page state no data licence. The home page describes "an open API for developers and researchers... at no charge". The footer "Terms of Use" links to a Cisco end-user licence agreement whose text was not available to read. No citation format is given. | Not verified. Read the linked Cisco agreement before publishing. Do not redistribute the data. Name PhishTank (Cisco Talos Intelligence Group) as the source with the access date. |
| Tranco | No licence for the list itself. It combines upstream providers with their own terms: Cisco Umbrella (free), Majestic (CC BY 3.0), CrUX (CC BY-SA 4.0), Cloudflare Radar (CC BY-NC 4.0, non-commercial), Farsight (default list only). | Academic, non-commercial use is consistent with these terms. Cite the paper below. |
| OpenPhish community feed | Terms of Use: for personal use; academic, independent or personal research allowed if it does not support an organisation's business operations; commercial use needs written consent; distributing or disclosing the information and creating derivative works are prohibited without written permission. The separate academic programme (60-day access, accredited institution email) requires citing OpenPhish with the access date and bars sharing. | Used for evaluation only, not redistributed, not trained on. Cite as `www.openphish.com`, accessed 2026-10-08. If the report ships any OpenPhish URLs, ask OpenPhish first. |

Citations for the report:
- Le Pochat, V., Van Goethem, T., Tajalizadehkhoob, S., Korczynski, M., and Joosen, W. (2019). Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation. NDSS 2019. https://doi.org/10.14722/ndss.2019.23386
- PhishTank, operated by Cisco Talos Intelligence Group. `data.phishtank.com/data/online-valid.csv.gz`, accessed 2026-10-08.
- OpenPhish community feed. `www.openphish.com`, accessed 2026-10-08.
