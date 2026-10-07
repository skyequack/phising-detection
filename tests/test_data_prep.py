import json

import pandas as pd
import pytest

from ml.data_prep import canonicalize, clean, drop_reason, run, split


def _raw(rows):
    return pd.DataFrame(rows, columns=["url", "type"])


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/a", None),
        ("http://192.168.1.1/login", None),
        ("https://x.com/?q=a b", None),  # inner whitespace is a real signal: kept
        ("", "empty_url"),
        ("http://muralove.net/x?rand\x13abc", "control_chars"),
        ("httpss://www.microsoft.com/", "bad_scheme"),
        ("ftp://ftp.cs.utexas.edu/pub/", "bad_scheme"),
        ("example.com/path", "bad_scheme"),
        ("http://ladiesfirst-privileges[.]com/a.exe", "unparseable"),
        ("http://?", "no_hostname"),
        ("http://#NAME?", "no_hostname"),
    ],
)
def test_drop_reason(url, expected):
    assert drop_reason(url) == expected


def test_canonicalize_ignores_scheme_www_slash_case():
    keys = {
        canonicalize(u)
        for u in [
            "https://www.Example.com/",
            "http://example.com",
            "HTTPS://EXAMPLE.COM//",
        ]
    }
    assert keys == {"example.com"}
    assert canonicalize("https://example.com/a") != canonicalize("https://example.com/b")


def test_clean_drops_artefacts_dupes_and_conflicts():
    raw = _raw(
        [
            ("https://good.com/a", "legitimate"),
            ("https://www.good.com/a/", "legitimate"),  # canonical duplicate
            ("https://evil.com/x", "phishing"),
            ("http://both.com", "legitimate"),  # conflict pair
            ("https://www.both.com/", "phishing"),
            ("http://?", "phishing"),  # no hostname
            ("httpss://bad.com", "legitimate"),  # bad scheme
            ("https://odd.com", "spam"),  # unknown label
            ("https://with space.com/a b", "phishing"),  # kept
        ]
    )
    clean_df, dropped, report = clean(raw)

    assert sorted(clean_df["url"]) == sorted(
        ["https://good.com/a", "https://evil.com/x", "https://with space.com/a b"]
    )
    assert report["dropped_by_reason"] == {
        "bad_label": 1,
        "bad_scheme": 1,
        "duplicate": 1,
        "label_conflict": 2,
        "no_hostname": 1,
    }
    assert report["label_conflict_groups"] == 1
    assert report["label_conflict_keys"] == ["both.com"]
    assert report["rows_in"] == len(raw)
    assert report["rows_clean"] + report["rows_dropped"] == len(raw)
    assert len(dropped) == report["rows_dropped"]
    assert set(clean_df["label"]) <= {0, 1}
    assert clean_df["label"].tolist().count(1) == 2  # phishing = 1


def test_clean_keeps_first_occurrence_of_duplicates():
    raw = _raw([("http://a.com/p", "phishing"), ("https://www.a.com/p/", "phishing")])
    clean_df, _, _ = clean(raw)
    assert clean_df["url"].tolist() == ["http://a.com/p"]


def test_split_is_stratified_disjoint_and_reproducible():
    df = pd.DataFrame(
        {
            "url": [f"https://site{i}.com" for i in range(1000)],
            "label": [1 if i % 4 == 0 else 0 for i in range(1000)],
        }
    )
    train, test = split(df, seed=7)
    assert len(train) == 800 and len(test) == 200
    assert train["label"].mean() == pytest.approx(0.25, abs=0.001)
    assert test["label"].mean() == pytest.approx(0.25, abs=0.001)
    assert set(train["url"]).isdisjoint(test["url"])
    train2, _ = split(df, seed=7)
    pd.testing.assert_frame_equal(train, train2)


def test_run_writes_outputs_and_leaves_input_untouched(tmp_path):
    rows = [(f"https://site{i}.com/p", "phishing" if i % 4 == 0 else "legitimate") for i in range(200)]
    rows += [("http://?", "phishing"), ("https://site1.com/p/", "legitimate")]  # artefact + dup
    src = tmp_path / "in.csv"
    _raw(rows).to_csv(src, index=False)
    before = src.read_bytes()

    report = run(src, tmp_path / "out", seed=1)

    assert src.read_bytes() == before
    for name in ["train.parquet", "test.parquet", "dropped.csv", "cleaning_report.json"]:
        assert (tmp_path / "out" / name).exists()
    saved = json.loads((tmp_path / "out" / "cleaning_report.json").read_text())
    assert saved["rows_in"] == 202
    assert saved["rows_clean"] == 200
    assert saved["train"]["rows"] + saved["test"]["rows"] == saved["rows_clean"]
    assert report["input_sha256"] == saved["input_sha256"]


def test_run_rejects_wrong_columns(tmp_path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"link": ["https://a.com"], "class": ["phishing"]}).to_csv(src, index=False)
    with pytest.raises(ValueError):
        run(src, tmp_path / "out")
