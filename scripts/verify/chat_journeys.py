"""Authored-fixture HTTP/worker smoke, including an explicit mock capability limit."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4
import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-sciq", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--output", default=str(ROOT / "evidence/integration/chat_journeys.json"))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("Choose a new evidence path; preserve earlier journeys")
    output.parent.mkdir(exist_ok=True, parents=True)
    client = httpx.Client(base_url=args.base_url, timeout=30)
    result = {
        "status": "running",
        "model_mode": "mock",
        "source_kind": "authored fixture",
        "scheduled_turns": 4,
        "turns": [],
        "scoring_or_sciq_used": False,
        "scope": "Software transport, context, citations and persistence; not semantic acceptance of practical-example generation.",
        "expected_outcomes": ["answer", "answer", "answer", "refusal"],
        "known_mock_limit": "The conservative extractive mock refuses the practical-example request. The supplied photosynthesis fixture contains no practical example; the mock does not synthesize one. Preserve this limitation rather than count it as a grounded example answer.",
    }

    def save():
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    save()

    def call(method, path, **kwargs):
        response = client.request(method, path, **kwargs)
        assert response.is_success, (method, path, response.status_code, response.text)
        return response.json()["data"]

    try:
        token = call(
            "POST", "/auth/login", json={"email": "student@example.com", "password": "Passw0rd!"}
        )["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        assert call("GET", "/capabilities")["model_mode"] == "mock", (
            "This smoke requires explicit mock answering"
        )
        session = call("POST", "/sessions", json={"title": "Verified no-SciQ chat"})
        result["session_id"] = session["id"]
        results = result["turns"]
        for prompt in [
            "What is photosynthesis?",
            "Why does it need light?",
            "Explain that more simply",
            "Give a practical example.",
        ]:
            key = str(uuid4())
            submission = call(
                "POST",
                f"/sessions/{session['id']}/messages",
                json={"content": prompt, "use_profile": True},
                headers={"Idempotency-Key": key},
            )
            duplicate = call(
                "POST",
                f"/sessions/{session['id']}/messages",
                json={"content": prompt, "use_profile": True},
                headers={"Idempotency-Key": key},
            )
            assert submission == duplicate
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = call("GET", "/jobs/" + submission["job_id"])
                if job["state"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.2)
            assert job["state"] == "succeeded", job
            answer = call("GET", "/answers/" + job["answer_id"])
            assert answer["response_schema"] == "chat_response_v1"
            assert answer["model_mode"] == "mock"
            expected = result["expected_outcomes"][len(results)]
            observed = {
                "prompt": prompt,
                "receipt": submission,
                "answer_id": answer["id"],
                "answer": answer["response"],
                "expected_response_type": expected,
                "context_message_ids": [
                    m["message_id"] for m in answer["conversation_snapshot"]["messages"]
                ],
            }
            result["last_observed_turn"] = observed
            save()
            assert answer["response"]["response_type"] == expected, answer["response"]
            if expected == "answer":
                assert answer["response"]["citations"]
            else:
                assert answer["response"]["refusal_reason"] == "INSUFFICIENT_EVIDENCE"
                assert answer["response"]["citations"] == []
                assert answer["response"]["short_answer"] is None
                assert "mock" in answer["response"]["answer_text"].lower()
            for cid in answer["response"]["citations"]:
                ev = call("GET", f"/answers/{answer['id']}/evidence/{cid}")
                assert hashlib.sha256(ev["text"].encode()).hexdigest() == ev["text_hash"]
            if len(results):
                assert answer["conversation_snapshot"]["messages"]
            results.append(observed)
            save()
            print(prompt, "=>", answer["response"]["response_type"], flush=True)
        feedback = call(
            "PUT",
            f"/answers/{answer['id']}/feedback",
            json={"helpful": True, "comment": "Verified during local integration test."},
        )
        assert call("GET", f"/answers/{answer['id']}/feedback")["id"] == feedback["id"]
        token = call(
            "POST", "/auth/login", json={"email": "student@example.com", "password": "Passw0rd!"}
        )["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        history = call("GET", f"/sessions/{session['id']}/messages")
        assert len(history["items"]) == 8
        assert [m["answer"]["id"] for m in history["items"] if m["role"] == "assistant"] == [
            r["answer_id"] for r in results
        ]
        result.update(
            status="passed_software_with_documented_mock_limit",
            answered_turns=3,
            refused_turns=1,
            history_after_relogin=len(history["items"]),
            feedback_id=feedback["id"],
        )
        save()
        print(
            "PASS: all 4 scheduled turns (3 cited answers, 1 explicit mock refusal), context, evidence hashes, idempotency, feedback and re-login history; no semantic example-generation claim."
        )
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        save()
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
