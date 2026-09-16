"""Create private qrel/blind-rating files and analyze actual independent reviews."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.annotations.blind import (
    analyze_ratings,
    blank_ratings_csv,
    create_blind_package,
    parse_ratings_csv,
)
from evaluation.annotations.qrels import create_template, validate_judgements
from evaluation.common import atomic_json, canonical, read_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qrel = sub.add_parser("qrels-template")
    qrel.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="JSON with corpus_release_id, processing_ids and question_candidates",
    )
    qrel.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("qrels-validate")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--release", required=True)
    check.add_argument("--processing", nargs="+", required=True)
    blind = sub.add_parser("blind-template")
    blind.add_argument("--outputs", type=Path, required=True)
    blind.add_argument("--destination", type=Path, required=True)
    blind.add_argument("--seed", type=int, default=0)
    rate = sub.add_parser("ratings-analyze")
    rate.add_argument("--package", type=Path, required=True)
    rate.add_argument("--ratings", type=Path, required=True)
    rate.add_argument("--reviewers", nargs="+", required=True)
    rate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "qrels-template":
        values = read_json(args.candidates)
        result = create_template(**values)
        atomic_json(args.output, result, immutable=True)
        print(canonical({"output": str(args.output), "unreviewed_rows": len(result["rows"])}))
    elif args.command == "qrels-validate":
        print(
            canonical(
                validate_judgements(
                    read_json(args.input),
                    corpus_release_id=args.release,
                    processing_ids=args.processing,
                )
            )
        )
    elif args.command == "blind-template":
        outputs = read_json(args.outputs)
        if isinstance(outputs, dict):
            outputs = outputs.get("data", outputs)
            if outputs.get("protocol_id") != "profile_study":
                raise ValueError("A study export must identify the separate profile_study protocol")
            outputs = outputs["items"]
        package = create_blind_package(outputs, seed=args.seed)
        args.destination.mkdir(parents=True, exist_ok=False)
        atomic_json(args.destination / "private_package.json", package, immutable=True)
        atomic_json(
            args.destination / "review_items.json",
            {"rubric": package["rubric"], "review_items": package["review_items"]},
            immutable=True,
        )
        (args.destination / "ratings.csv").write_text(
            blank_ratings_csv(package), encoding="utf-8", newline=""
        )
        print(
            canonical(
                {
                    "scheduled_outputs": package["scheduled_count"],
                    "destination": str(args.destination),
                    "actual_ratings": 0,
                }
            )
        )
    else:
        package = read_json(args.package)
        ratings = parse_ratings_csv(args.ratings.read_text(encoding="utf-8-sig"), package)
        result = analyze_ratings(package, ratings, expected_reviewers=args.reviewers)
        atomic_json(args.output, result)
        print(
            canonical(
                {
                    "output": str(args.output),
                    "received_rating_rows": len(ratings),
                    "learning_gain_measured": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
