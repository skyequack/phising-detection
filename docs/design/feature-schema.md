# Feature schema (v1)

Implemented in `ml/features.py` (`extract_features`, `extract_matrix`), tested in `tests/test_features.py`. 30 URL-only features from report Table 3.2, in this column order. Standard library only; no DNS or HTTP. Inputs over 4,096 characters or non-strings raise `ValueError`. Malformed URLs never crash: if `urlsplit` raises, a regex split is used instead.

Host = `urlsplit(url).hostname` (lowercased, userinfo and port removed). The URL is `strip()`ped but otherwise used as given (no decoding).

| Group | Feature | Definition |
|---|---|---|
| length | `url_length`, `hostname_length`, `path_length`, `query_length` | Character counts of the URL, host, path, query |
| length | `path_to_url_ratio` | `path_length / url_length` |
| char_count | `num_digits`, `num_letters` | ASCII digits; Unicode letters |
| char_count | `num_special_chars` | Non-alphanumeric characters (whitespace and non-ASCII punctuation included) |
| char_count | `num_dots`, `num_hyphens`, `num_slashes` | Counts over the whole URL |
| char_count | `digit_ratio`, `special_char_ratio` | Count / `url_length` |
| host_structure | `num_subdomains` | Host labels minus registrable domain (2 labels, or 3 under `co.uk`-style suffixes; small built-in list, no public-suffix database). 0 for IP hosts |
| host_structure | `has_ip_host` | IPv4/IPv6, all-digit (`3232235777`) or `0x...` host |
| host_structure | `has_at_symbol` / `has_userinfo` | `@` anywhere / `@` in the authority |
| host_structure | `has_port` | Explicit `:digits` after the host |
| host_structure | `longest_host_token` | Longest dot-separated host label |
| host_structure | `has_punycode` | Any host label starts with `xn--` |
| query_structure | `has_query`, `has_fragment` | Non-empty query; `#` present |
| query_structure | `num_query_params` | Non-empty `&`-separated pairs |
| query_structure | `num_encoded_chars` | `%XX` occurrences |
| query_structure | `has_double_slash_path` | `//` inside the path |
| query_structure | `longest_path_segment` | Longest `/`-separated path segment |
| query_structure | `has_hex_token` | A standalone token of 8+ hex chars with at least one digit and one a-f letter |
| entropy | `hostname_entropy`, `url_entropy` | Shannon entropy (bits/char) |
| keywords | `suspicious_keyword_count` | Distinct keywords from `SUSPICIOUS_KEYWORDS` found as substrings (case-insensitive) |

## Decisions (resolves part of the open ablation-group question)

- `path_to_url_ratio` goes in **length** (it is a length ratio). `longest_host_token` and `has_punycode` go in **host_structure**. `longest_path_segment`, `has_hex_token`, `has_double_slash_path` and `num_encoded_chars` have no obvious home and are grouped under **query_structure**, which therefore covers path and query. Revisit before the ablation study.
- `num_subdomains` uses a heuristic suffix list rather than adding a `tldextract` dependency. Unusual suffixes (e.g. `.com.au` is handled, `.pvt.k12.ma.us` is not) can be off by one.
- Whitespace is not stripped inside the URL. All 521 whitespace URLs in the raw data are phishing, so `num_special_chars` can leak that label; mention as a limitation.

## Measured on the training split (350,141 rows, 2026-10-07)

No NaNs; about 31 s (~90 µs/URL). Class means differ in the expected direction for digits, keywords, IP hosts, `@`, hex tokens and query length. Two collection-bias signals to cite as limitations: legitimate URLs have *more* subdomains (1.26 vs 0.37) and more hyphens (1.46 vs 0.57) than phishing, and `has_fragment` is almost never set for legitimate URLs (0.000 vs 0.004). Check these in the ablation study.
