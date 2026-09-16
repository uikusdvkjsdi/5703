"""Derive a hash-bound publication summary from a completed full source check."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(path):
    report = json.loads(path.read_text("utf-8"))
    if report["status"] != "passed" or report["errors"]:
        raise ValueError("The complete source check must pass before summarization")
    books = []
    for book in report["books"]:
        counts = Counter(code for page in book["pages"] for code in page["issue_codes"])
        books.append(
            {
                "slug": book["slug"],
                "processing_id": book["processing_id"],
                "counts": book["counts"],
                "chunks": book["chunk_count"],
                "physical_pages": book["physical_page_count"],
                "max_body_tokens": book["maximum_body_tokens"],
                "max_input_tokens": book["maximum_input_tokens"],
                "issue_page_counts": dict(counts),
                "sample_count": sum(item["slug"] == book["slug"] for item in report["sample"]),
            }
        )
    return {
        "status": report["status"],
        "formal_report": path.as_posix(),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validator_script_sha256": hashlib.sha256(
            Path(__file__).with_name("formal_corpus.py").read_bytes()
        ).hexdigest(),
        "total_physical_pages": sum(book["physical_pages"] for book in books),
        "total_units": sum(book["counts"]["total_units"] for book in books),
        "total_chunks": sum(book["chunks"] for book in books),
        "max_body_tokens": max(book["max_body_tokens"] for book in books),
        "max_input_tokens": max(book["max_input_tokens"] for book in books),
        "sample_count": len(report["sample"]),
        "sample_original_pages_checked": len(
            {
                (item["slug"], page["physical_page"])
                for item in report["sample"]
                for page in item["original_pdf_checks"]
            }
        ),
        "books": books,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Existing summaries are preserved; choose a new output")
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "chunks": result["total_chunks"],
                "source_sha256": result["report_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
