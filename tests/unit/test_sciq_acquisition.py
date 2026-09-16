"""Lossless private acquisition and public projection guards using authored rows."""

import json
from pathlib import Path

import pytest

from evaluation.datasets.sciq import load_split
from scripts.dev.acquire_sciq import acquire, sha256
from scripts.verify.sciq import verify


def fixture_source(monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = [
        {
            "question": "Where is oxygen produced?",
            "correct_answer": "chloroplast",
            "distractor1": "nucleus",
            "distractor2": "nucleus",
            "distractor3": "ribosome",
            "support": "PRIVATE_SUPPORT\u2028with a Unicode separator",
        }
    ]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(source), sink)
    raw = sink.getvalue().to_pybytes()
    revision = "a" * 40
    config = {
        "repository": "allenai/sciq",
        "revision": revision,
        "license": "cc-by-nc-3.0",
        "source_homepage": "https://allenai.org/data/sciq",
        "paper_url": "https://aclanthology.org/W17-4413/",
        "splits": {
            "validation": {
                "count": 1,
                "path": "data/validation.parquet",
                "size_bytes": len(raw),
                "sha256": sha256(raw),
            }
        },
    }
    metadata = {
        "sha": revision,
        "id": "allenai/sciq",
        "lastModified": "source-fixture-date",
        "cardData": {
            "license": ["cc-by-nc-3.0"],
            "dataset_info": {"splits": [{"name": "validation", "num_examples": 1}]},
        },
    }

    def download(url, maximum=10_000_000):
        if "/api/" in url:
            return json.dumps(metadata).encode()
        if url.endswith("README.md"):
            return b"Authored source-card fixture"
        return raw

    monkeypatch.setattr("scripts.dev.acquire_sciq.download", download)
    return source, config


def test_lossless_parquet_unicode_and_all_item_public_preflight_boundary(tmp_path, monkeypatch):
    source, config = fixture_source(monkeypatch)
    manifest = acquire(config, tmp_path / "private", tmp_path / "acquisition.json")
    derived = Path(manifest["splits"][0]["derived_path"])
    rows, _ = load_split(
        derived, split="validation", revision=config["revision"], require_distinct_choices=False
    )
    assert rows[0].support == source[0]["support"]
    assert "\\u2028" in derived.read_text()
    report = verify(manifest, tmp_path / "public", tmp_path / "verification.json")
    assert report["total_native_records"] == report["total_openqa_ready"] == 1
    assert report["total_mcq_ready"] == 0 and report["total_mcq_input_rejected"] == 1
    public_files = list((tmp_path / "public").rglob("*.jsonl"))
    assert public_files
    for file in public_files:
        text = file.read_text()
        assert "PRIVATE_SUPPORT" not in text and "chloroplast" not in text
        assert "correct_label" not in text and "reference" not in text
    assert "PRIVATE_SUPPORT" not in (tmp_path / "acquisition.json").read_text()
    assert "PRIVATE_SUPPORT" not in (tmp_path / "verification.json").read_text()
    assert (
        "PRIVATE_SUPPORT"
        in (
            Path(manifest["private_storage_root"]) / "projections/validation.references.jsonl"
        ).read_text()
    )
    # Rechecking exact bytes is idempotent and preserves the first evidence date.
    assert verify(manifest, tmp_path / "public", tmp_path / "verification.json") == report
    with pytest.raises(ValueError, match="distinct"):
        verify(manifest, Path(manifest["private_storage_root"]) / "public", tmp_path / "bad.json")


def test_official_byte_hash_mismatch_stops_before_raw_publication(tmp_path, monkeypatch):
    _, config = fixture_source(monkeypatch)
    config["splits"]["validation"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned LFS"):
        acquire(config, tmp_path / "private", tmp_path / "acquisition.json")
    assert not list((tmp_path / "private").rglob("*.parquet"))
    assert not (tmp_path / "acquisition.json").exists()
