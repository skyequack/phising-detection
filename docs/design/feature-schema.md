# Feature schema (v1)

Implemented in `ml/features.py` (`extract_features`, `extract_matrix`), tested in `tests/test_features.py`. 30 URL-only features from report Table 3.2, in this column order. Standard library only; no DNS or HTTP. Empty, non-string or over-4,096-character input raises `InvalidURLError` (a `ValueError` subclass the API maps to 422). Malformed URLs never crash: if `urlsplit` raises, a regex split is used instead.

Host = `urlsplit(url).hostname` (lowercased, userinfo and port removed). The URL is `strip()`ped but otherwise used as given (no decoding).

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

## Measured on the training split (350,141 rows, 2026-10-07)

No NaNs; about 31 s (~90 µs/URL). Class means differ in the expected direction for digits, keywords, IP hosts, `@`, hex tokens and query length. Two collection-bias signals to cite as limitations: legitimate URLs have *more* subdomains (1.26 vs 0.37) and more hyphens (1.46 vs 0.57) than phishing, and `has_fragment` is almost never set for legitimate URLs (0.000 vs 0.004). Check these in the ablation study.
