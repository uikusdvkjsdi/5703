"""Acquire a fixed official SciQ snapshot into evaluator-only private storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"question", "correct_answer", "support", "distractor1", "distractor2", "distractor3"}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Existing immutable artifact differs: {path.name}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def download(url: str, maximum: int = 10_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CS30-source-acquisition/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Official response exceeds the bounded expected download size")
    return raw


def acquire(config: dict, private_root: Path, evidence_path: Path) -> dict:
    import pyarrow.parquet as pq

    revision = config["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An immutable full dataset revision is required")
    if config["repository"] != "allenai/sciq":
        raise ValueError("Only the verified research-owner repository is supported")
    directory = private_root.resolve() / "sciq" / revision
    api_url = f"https://huggingface.co/api/datasets/allenai/sciq/revision/{revision}"
    metadata_path = directory / "source-metadata.json"
    metadata_bytes = metadata_path.read_bytes() if metadata_path.exists() else download(api_url)
    metadata = json.loads(metadata_bytes)
    if metadata["sha"] != revision or metadata["id"] != config["repository"]:
        raise ValueError("Research-owner metadata identity differs from the pinned revision")
    if config["license"] not in metadata["cardData"]["license"]:
        raise ValueError("Dataset license differs from the pinned acquisition record")
    immutable(metadata_path, metadata_bytes)
    base = f"https://huggingface.co/datasets/allenai/sciq/resolve/{revision}/"
    card_path = directory / "README.source.md"
    immutable(
        card_path, card_path.read_bytes() if card_path.exists() else download(base + "README.md")
    )
    counts = {
        row["name"]: row["num_examples"] for row in metadata["cardData"]["dataset_info"]["splits"]
    }
    splits = []
    for split, expected in config["splits"].items():
        if counts[split] != expected["count"]:
            raise ValueError("Publisher split count differs from the pinned configuration")
        path = directory / "raw" / Path(expected["path"]).name
        raw = path.read_bytes() if path.exists() else download(base + expected["path"])
        if len(raw) != expected["size_bytes"] or sha256(raw) != expected["sha256"]:
            raise ValueError(f"Official {split} bytes differ from their pinned LFS SHA-256")
        immutable(path, raw)
        table = pq.read_table(path)
        if set(table.column_names) != FIELDS or table.num_rows != expected["count"]:
            raise ValueError("Original Parquet fields/count are incompatible with the pinned split")
        rows = table.to_pylist()
        if any(
            set(row) != FIELDS or any(not isinstance(v, str) for v in row.values()) for row in rows
        ):
            raise ValueError("SciQ native records must contain exactly six text fields")
        # Preserve source order, Unicode and every field value exactly. No label
        # correction, filtering, support injection, deduplication or text cleanup.
        # Escape Unicode separators in JSON syntax so generic JSONL readers
        # cannot confuse U+0085/U+2028/U+2029 inside source strings with rows.
        # Decoding still reproduces the exact original Unicode field values.
        converted = "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
        converted_path = directory / "derived" / "json_ascii_v1" / f"{split}.jsonl"
        immutable(converted_path, converted)
        if [json.loads(line) for line in converted.decode("utf-8").splitlines()] != rows:
            raise ValueError("Private JSONL conversion changed original records")
        splits.append(
            {
                "split": split,
                "count": len(rows),
                "source_url": base + expected["path"],
                "raw_path": str(path),
                "raw_sha256": sha256(raw),
                "raw_bytes": len(raw),
                "derived_path": str(converted_path),
                "derived_sha256": sha256(converted),
                "derived_bytes": len(converted),
                "empty_support_count": sum(not row["support"].strip() for row in rows),
                "source_fields_and_order_preserved": True,
            }
        )
    manifest_path = directory / "acquisition.json"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result["splits"] != splits:
            raise ValueError("Reacquisition differs from the immutable source manifest")
    else:
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": "SciQ",
            "repository": config["repository"],
            "revision": revision,
            "repository_last_modified": metadata["lastModified"],
            "license": config["license"],
            "source_homepage": config["source_homepage"],
            "paper_url": config["paper_url"],
            "source_metadata_sha256": sha256(metadata_bytes),
            "source_card_sha256": sha256(card_path.read_bytes()),
            "private_storage_root": str(directory),
            "converter": {
                "pyarrow": importlib.metadata.version("pyarrow"),
                "operation": "Parquet rows to lossless JSONL in original split order",
            },
            "splits": splits,
            "total_records": sum(row["count"] for row in splits),
            "model_calls": 0,
            "corpus_ingestion": False,
            "source_notice": "The source owner identifies CC BY-NC 3.0. Source card, original bytes and attribution are retained; no claim of scientific correctness or model accuracy is made.",
        }
        immutable(manifest_path, json.dumps(result, indent=2).encode("utf-8"))
    # Contains only identities, counts, paths and hashes; no questions, options,
    # private support, correct answers, reviewer keys or per-item gold labels.
    immutable(evidence_path, json.dumps(result, indent=2).encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/sciq-source.json")
    parser.add_argument(
        "--private-root",
        default=os.environ.get("EVALUATION_PRIVATE_ROOT", "artifacts/evaluator-private"),
    )
    parser.add_argument("--evidence", default="evidence/sciq/acquisition.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = acquire(config, Path(args.private_root), Path(args.evidence))
    print(
        json.dumps(
            {
                "revision": result["revision"],
                "count": result["total_records"],
                "splits": [
                    {"split": s["split"], "count": s["count"], "raw_sha256": s["raw_sha256"]}
                    for s in result["splits"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
