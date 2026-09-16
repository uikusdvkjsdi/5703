"""Version qrels against exact released source spans and judged pools."""

from __future__ import annotations

from evaluation.common import fingerprint, utc_now


def create_template(
    *,
    corpus_release_id: str,
    processing_ids: list[str],
    question_candidates: dict[str, list[dict]],
    version: str = "qrels-v1",
) -> dict:
    if not corpus_release_id or not processing_ids or not version:
        raise ValueError("Qrels need a named release, processing identities and version")
    rows = []
    for question_id, candidates in question_candidates.items():
        seen = set()
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            if chunk_id in seen:
                raise ValueError("A qrel pool cannot duplicate a question/chunk pair")
            seen.add(chunk_id)
            if candidate["processing_id"] not in processing_ids:
                raise ValueError("Candidate uses an incompatible processing identity")
            rows.append(
                {
                    "question_id": question_id,
                    "chunk_id": chunk_id,
                    "processing_id": candidate["processing_id"],
                    "text_hash": candidate["text_hash"],
                    "source_spans": candidate["source_spans"],
                    "text": candidate["text"],
                    "grade": None,
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "notes": "",
                }
            )
    identity = {
        "version": version,
        "corpus_release_id": corpus_release_id,
        "processing_ids": sorted(processing_ids),
        "pool": [
            {
                key: row[key]
                for key in ("question_id", "chunk_id", "processing_id", "text_hash", "source_spans")
            }
            for row in rows
        ],
    }
    return {
        "identity": identity,
        "pool_hash": fingerprint(identity),
        "created_at": utc_now(),
        "rows": rows,
        "status": "unreviewed",
    }


def validate_judgements(
    template: dict, *, corpus_release_id: str, processing_ids: list[str]
) -> dict:
    identity = template["identity"]
    if (
        identity["corpus_release_id"] != corpus_release_id
        or identity["processing_ids"] != sorted(processing_ids)
        or fingerprint(identity) != template["pool_hash"]
    ):
        raise ValueError("Qrels are incompatible with the selected corpus or processing release")
    expected = {(row["question_id"], row["chunk_id"]): row for row in identity["pool"]}
    seen, results = set(), {}
    import hashlib

    for row in template["rows"]:
        key = (row["question_id"], row["chunk_id"])
        if (
            key in seen
            or key not in expected
            or any(
                row[field] != expected[key][field]
                for field in ("processing_id", "text_hash", "source_spans")
            )
        ):
            raise ValueError("Judged pool or source identity was changed without a new version")
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_hash"]:
            raise ValueError("Judgement text does not match the frozen source hash")
        seen.add(key)
        grade = row.get("grade")
        if grade is not None and (
            isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3
        ):
            raise ValueError("Relevance grade must be 0–3 or unavailable")
        if grade is not None and (not row.get("reviewer_id") or not row.get("reviewed_at")):
            raise ValueError("Actual judgements need reviewer identity and review time")
        results.setdefault(key[0], {})[key[1]] = grade
    if seen != set(expected):
        raise ValueError("Unjudged candidates must remain with null grades, not disappear")
    return {
        "qrels": results,
        "judged_count": sum(
            grade is not None for values in results.values() for grade in values.values()
        ),
        "scheduled_judgements": len(seen),
        "pool_hash": template["pool_hash"],
    }


def compare_chunk_sets(before: list[dict], after: list[dict]) -> dict:
    old = {row["chunk_id"]: row for row in before}
    new = {row["chunk_id"]: row for row in after}
    common = set(old) & set(new)
    changed = sorted(
        key
        for key in common
        if old[key]["text_hash"] != new[key]["text_hash"]
        or old[key]["source_spans"] != new[key]["source_spans"]
    )
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": changed,
        "unchanged": sorted(common - set(changed)),
        "requires_qrel_review": bool(set(old) != set(new) or changed),
        "mapping_policy": "Source-span overlap suggests candidates for human review; no relevance grade is transferred automatically.",
    }
