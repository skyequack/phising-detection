"""Phase 2: URL-only feature extraction (report Table 3.2, 30 features).

``extract_features(url)`` is the single source of truth, used by training and
by the API. Pure and deterministic, standard library only: no DNS, no HTTP.
Malformed input never crashes the parser; it falls back to a regex split.
Empty, non-string or over-``MAX_URL_LENGTH`` input raises ``InvalidURLError`` (a ``ValueError``) (the API turns that
into a validation error).
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlsplit

import pandas as pd

MAX_URL_LENGTH = 4096
SCHEMA_VERSION = "2"

# Ordered as in Table 3.2; the model's column order is this order.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "length": ("url_length", "hostname_length", "path_length", "query_length"),
    "char_count": (
        "num_digits", "num_letters", "num_special_chars", "num_dots", "num_hyphens",
        "num_slashes", "digit_ratio", "special_char_ratio",
    ),
    "host": (
        "num_subdomains", "has_ip_host", "has_port", "has_userinfo", "has_at_symbol",
        "longest_host_token", "has_punycode",
    ),
    "query_path": (
        "has_query", "num_query_params", "num_encoded_chars", "has_fragment",
        "has_double_slash_path", "path_to_url_ratio", "longest_path_segment",
    ),
    "entropy": ("hostname_entropy", "url_entropy"),
    "keyword": ("suspicious_keyword_count", "has_hex_token"),
}
FEATURE_NAMES: tuple[str, ...] = tuple(n for g in FEATURE_GROUPS.values() for n in g)
assert len(FEATURE_NAMES) == 30 and len(set(FEATURE_NAMES)) == 30

SUSPICIOUS_KEYWORDS = (
    "login", "verify", "secure", "account", "update", "bank", "confirm", "signin", "password",
)


class InvalidURLError(ValueError):
    """Input rejected before parsing (empty, non-string, or too long); the API maps it to 422."""


# Second-level labels that, under a 2-letter ccTLD, form a public suffix (co.uk).
_SLD = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"}
_SPLIT_FALLBACK = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?(?P<netloc>[^/?#]*)(?P<rest>.*)$", re.I | re.S)
_PORT = re.compile(r"(?::|\]:)(\d+)$")
_ENCODED = re.compile(r"%[0-9a-fA-F]{2}")
_HEX_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}(?![A-Za-z0-9])", re.I)
_HEX_IP = re.compile(r"^0x[0-9a-f]+$", re.I)


_ANY_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_HTTP = re.compile(r"^https?://", re.I)
_WWW = re.compile(r"^www\.", re.I)
_BARE_SLASH = re.compile(r"^(http://[^/?#]*)/$")


def normalize_url(url: str) -> str:
    """Remove dataset-shortcut signals before feature extraction.

    In the training data 100% of legitimate URLs are https and 99.8% start with
    ``www.`` (phishing: 6.4% and 8.1%), so the scheme, a leading ``www.`` and a
    lone trailing ``/`` would let a model separate the classes without learning
    anything about phishing. Every URL is given the scheme ``http://``, loses a
    leading ``www.`` and a bare trailing slash. Other schemes are left untouched.
    """
    if _ANY_SCHEME.match(url) and not _HTTP.match(url):
        return url
    rest = _WWW.sub("", _HTTP.sub("", url))
    return _BARE_SLASH.sub(lambda m: m.group(1), "http://" + rest)


def _split(url: str) -> tuple[str, str, str, str, str]:
    """Return (netloc, host, path, query, fragment); never raises."""
    try:
        p = urlsplit(url)
        return p.netloc, (p.hostname or ""), p.path, p.query, p.fragment
    except ValueError:  # e.g. 'http://bad[.]com/x'
        m = _SPLIT_FALLBACK.match(url)
        netloc, rest = m["netloc"], m["rest"]
        host = netloc.rsplit("@", 1)[-1]
        host = _PORT.sub("", host).strip("[]").lower()
        rest, _, fragment = rest.partition("#")
        path, _, query = rest.partition("?")
        return netloc, host, path, query, fragment


def _is_ip(host: str) -> bool:
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host.isdigit() or bool(_HEX_IP.match(host))  # 3232235777 / 0xC0A80001


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _num_subdomains(host: str) -> int:
    if not host or _is_ip(host):
        return 0
    labels = host.split(".")
    registrable = 3 if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SLD else 2
    return max(0, len(labels) - registrable)


def extract_features(url: str) -> dict[str, float]:
    """Compute the 30 Table 3.2 features for one URL, in ``FEATURE_NAMES`` order.

    Features are computed on ``normalize_url(url)`` (see there).
    """
    if not isinstance(url, str):
        raise InvalidURLError("url must be a string")
    url = url.strip()
    if not url:
        raise InvalidURLError("url is empty")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError(f"url longer than {MAX_URL_LENGTH} characters")
    url = normalize_url(url)

    netloc, host, path, query, fragment = _split(url)
    n = len(url)
    digits = sum("0" <= c <= "9" for c in url)
    letters = sum(c.isalpha() for c in url)
    special = sum(not c.isalnum() for c in url)
    lower = url.lower()
    labels = host.split(".") if host else []

    f = {
        "url_length": n,
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "path_to_url_ratio": len(path) / n if n else 0.0,
        "num_digits": digits,
        "num_letters": letters,
        "num_special_chars": special,
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "digit_ratio": digits / n if n else 0.0,
        "special_char_ratio": special / n if n else 0.0,
        "num_subdomains": _num_subdomains(host),
        "has_ip_host": int(_is_ip(host)),
        "has_at_symbol": int("@" in url),
        "has_port": int(bool(_PORT.search(netloc.rsplit("@", 1)[-1]))),
        "has_userinfo": int("@" in netloc),
        "longest_host_token": max(map(len, labels), default=0),
        "has_punycode": int(any(l.startswith("xn--") for l in labels)),
        "has_fragment": int("#" in url),
        "has_query": int(bool(query)),
        "num_query_params": sum(1 for kv in query.split("&") if kv),
        "num_encoded_chars": len(_ENCODED.findall(url)),
        "has_double_slash_path": int("//" in path),
        "longest_path_segment": max(map(len, path.split("/")), default=0),
        "has_hex_token": int(bool(_HEX_TOKEN.search(url))),
        "hostname_entropy": _entropy(host),
        "url_entropy": _entropy(url),
        "suspicious_keyword_count": sum(k in lower for k in SUSPICIOUS_KEYWORDS),
    }
    return {name: f[name] for name in FEATURE_NAMES}


def extract_matrix(urls: Iterable[str]) -> pd.DataFrame:
    """Feature table (one row per URL, columns in ``FEATURE_NAMES`` order)."""
    return pd.DataFrame([extract_features(u) for u in urls], columns=list(FEATURE_NAMES))
