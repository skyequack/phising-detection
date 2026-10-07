# Feature schema (v2)

Implemented in `ml/features.py` (`extract_features`, `extract_matrix`), tested in `tests/test_features.py`. 30 URL-only features from report Table 3.2, in this column order. Standard library only; no DNS or HTTP. Empty, non-string or over-4,096-character input raises `InvalidURLError` (a `ValueError` subclass the API maps to 422). Malformed URLs never crash: if `urlsplit` raises, a regex split is used instead.

Host = `urlsplit(url).hostname` (lowercased, userinfo and port removed). The URL is `strip()`ped and then normalized by `normalize_url` (no percent-decoding).

## Normalization (added in schema v2)

Every `http://` or `https://` URL, or scheme-less input, is rewritten before any feature is computed:

1. Scheme becomes `http://`.
2. A leading `www.` (case-insensitive) is removed from the host.
3. A lone trailing `/` after the host (`http://host/`) is removed.

URLs with other schemes are left as they are. `www.` elsewhere in the host (`www.paypal.com.evil.example` becomes `paypal.com.evil.example`, but `a.www.example.com` keeps its `www`) is a real signal and is kept.

Why: in the training data 100% of legitimate URLs are `https` and 99.8% start with `www.` (phishing: 6.4% and 8.1%), and the scheme and `www.` change `url_length`, `hostname_length`, `num_dots`, `num_slashes` and `num_subdomains`. Without normalization the models separated the classes by those artefacts (see `docs/dataset/README.md`, "Shortcut signals"). Consequence: lengths and counts describe the normalized URL, and http versus https carries no signal.

| Group | Feature | Definition |
|---|---|---|
| length | `url_length`, `hostname_length`, `path_length`, `query_length` | Character counts of the URL, host, path, query |
| char_count | `num_digits`, `num_letters` | ASCII digits; Unicode letters |
| char_count | `num_special_chars` | Non-alphanumeric characters (whitespace and non-ASCII punctuation included) |
| char_count | `num_dots`, `num_hyphens`, `num_slashes` | Counts over the whole URL |
| char_count | `digit_ratio`, `special_char_ratio` | Count / `url_length` |
| host | `num_subdomains` | Host labels minus registrable domain (2 labels, or 3 under `co.uk`-style suffixes; small built-in list, no public-suffix database). 0 for IP hosts |
| host | `has_ip_host` | IPv4/IPv6, all-digit (`3232235777`) or `0x...` host |
| host | `has_at_symbol` / `has_userinfo` | `@` anywhere / `@` in the authority |
| host | `has_port` | Explicit `:digits` after the host |
| host | `longest_host_token` | Longest dot-separated host label |
| host | `has_punycode` | Any host label starts with `xn--` |
| query_path | `has_query`, `has_fragment` | Non-empty query; `#` present |
| query_path | `path_to_url_ratio` | `path_length / url_length` |
| query_path | `num_query_params` | Non-empty `&`-separated pairs |
| query_path | `num_encoded_chars` | `%XX` occurrences |
| query_path | `has_double_slash_path` | `//` inside the path |
| query_path | `longest_path_segment` | Longest `/`-separated path segment |
| keyword | `has_hex_token` | A standalone token of 8+ hex chars with at least one digit and one a-f letter |
| entropy | `hostname_entropy`, `url_entropy` | Shannon entropy (bits/char) |
| keyword | `suspicious_keyword_count` | Distinct keywords from `SUSPICIOUS_KEYWORDS` (login verify secure account update bank confirm signin password) found as substrings, case-insensitive |

## Decisions (resolves part of the open ablation-group question)

- Group assignment follows the implementation plan (length 4, char_count 8, host 7, query_path 7, entropy 2, keyword 2).
- `num_subdomains` uses a heuristic suffix list rather than adding a `tldextract` dependency. Unusual suffixes (e.g. `.com.au` is handled, `.pvt.k12.ma.us` is not) can be off by one.
- Whitespace is not stripped inside the URL. All 521 whitespace URLs in the raw data are phishing, so `num_special_chars` can leak that label; mention as a limitation.

## Measurements

- Extraction speed (schema v1, before normalization): about 31 s for the 350,141-row training split (about 90 microseconds per URL), no NaNs. Not re-measured for v2; normalization adds a few regex operations per URL.
- Feature-group importance in the selected Random Forest (`metadata.json`): char_count 0.384, query_path 0.174, entropy 0.141, length 0.118, keyword 0.093, host 0.090.
- Ablation on the test split (`reports/metrics/ablation.csv`): dropping char_count costs the most (F1 -0.052), then host (-0.027), entropy (-0.015), keyword (-0.014), query_path (-0.008) and length (-0.002).
- Class means of individual features are not reproduced here; regenerate them from `data/processed/features_*_train.parquet` if the report needs them. In the v1 data, legitimate URLs had more subdomains and hyphens than phishing (mostly the `www.` artefact), so treat raw class means with care.
