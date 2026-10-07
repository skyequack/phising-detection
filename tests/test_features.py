import pytest

from ml.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    MAX_URL_LENGTH,
    extract_features,
    extract_matrix,
)

BASIC = "https://www.example.com/a/b?x=1&y=2#top"


def test_schema_has_30_unique_features_in_six_groups():
    assert len(FEATURE_NAMES) == 30 and len(set(FEATURE_NAMES)) == 30
    assert len(FEATURE_GROUPS) == 6


def test_output_order_and_determinism():
    a, b = extract_features(BASIC), extract_features(BASIC)
    assert list(a) == list(FEATURE_NAMES)
    assert a == b


def test_basic_values():
    f = extract_features(BASIC)
    assert f["url_length"] == len(BASIC)
    assert f["hostname_length"] == len("www.example.com")
    assert f["path_length"] == len("/a/b")
    assert f["query_length"] == len("x=1&y=2")
    assert f["num_subdomains"] == 1
    assert f["has_query"] == 1 and f["num_query_params"] == 2
    assert f["has_fragment"] == 1
    assert f["num_slashes"] == 4
    assert f["num_digits"] == 2
    assert f["longest_host_token"] == len("example")
    assert f["longest_path_segment"] == 1
    assert f["has_ip_host"] == f["has_at_symbol"] == f["has_port"] == f["has_userinfo"] == 0


@pytest.mark.parametrize(
    "url,host_sub",
    [
        ("http://example.com", 0),
        ("http://a.b.example.com/", 2),
        ("http://shop.example.co.uk/", 1),
        ("http://192.168.0.1/login", 0),
    ],
)
def test_num_subdomains(url, host_sub):
    assert extract_features(url)["num_subdomains"] == host_sub


@pytest.mark.parametrize("url", ["http://192.168.0.1/x", "http://3232235777/", "http://0xC0A80001/", "http://[::1]/"])
def test_ip_hosts(url):
    assert extract_features(url)["has_ip_host"] == 1


def test_userinfo_port_at():
    f = extract_features("http://paypal.com@evil.example:8080/x")
    assert f["has_at_symbol"] == f["has_userinfo"] == f["has_port"] == 1
    assert f["hostname_length"] == len("evil.example")
    # '@' only in the path: at-symbol yes, userinfo no
    g = extract_features("http://example.com/a@b")
    assert g["has_at_symbol"] == 1 and g["has_userinfo"] == 0


def test_punycode_hex_encoded_double_slash():
    f = extract_features("http://xn--pple-43d.com//a%20b/deadbeef12?q=%41")
    assert f["has_punycode"] == 1
    assert f["has_double_slash_path"] == 1
    assert f["num_encoded_chars"] == 2
    assert f["has_hex_token"] == 1
    assert extract_features("http://example.com/12345678")["has_hex_token"] == 0
    assert extract_features("http://example.com/abcdefgh")["has_hex_token"] == 0


def test_keywords_and_entropy():
    f = extract_features("http://secure-login.example.com/verify/account")
    assert f["suspicious_keyword_count"] == 4  # secure, login, verify, account
    assert extract_features("http://aaaa/")["hostname_entropy"] == 0.0
    assert extract_features("http://abcd/")["hostname_entropy"] == pytest.approx(2.0)


def test_ratios_bounded_and_correct():
    f = extract_features("http://a1.com/9")
    assert f["digit_ratio"] == pytest.approx(2 / len("http://a1.com/9"))
    assert 0 <= f["special_char_ratio"] <= 1 and 0 <= f["path_to_url_ratio"] <= 1


@pytest.mark.parametrize(
    "url",
    [
        "http://ladiesfirst-privileges[.]com/x",  # urlsplit raises ValueError
        "http://#NAME?",
        "http://?",
        "",
        "   ",
        "not a url",
        "http://a\x13b.com",
        "http://[::1",
        "http://example.com:notaport/",
        'http://x.com/a.php""></script><script>',
        "http://x.com/' OR 1=1 --",
        "http://münchen.de/ü",
    ],
)
def test_malformed_input_never_crashes_and_stays_finite(url):
    f = extract_features(url)
    assert list(f) == list(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf") for v in f.values())


def test_bracket_fallback_still_finds_host():
    f = extract_features("http://ladiesfirst-privileges[.]com/x?a=1")
    assert f["hostname_length"] > 0 and f["has_query"] == 1


def test_length_limit_and_type():
    extract_features("http://x.com/" + "a" * (MAX_URL_LENGTH - 13))
    with pytest.raises(ValueError):
        extract_features("http://x.com/" + "a" * MAX_URL_LENGTH)
    with pytest.raises(ValueError):
        extract_features(None)  # type: ignore[arg-type]


def test_no_network(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    extract_features("http://example.com/a")


def test_extract_matrix_shape_and_order():
    df = extract_matrix([BASIC, "http://1.2.3.4/"])
    assert df.shape == (2, 30) and list(df.columns) == list(FEATURE_NAMES)
