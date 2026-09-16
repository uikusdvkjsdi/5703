"""Exercise real HTTP/worker/E5/pgvector flows while keeping answers explicitly mock."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import httpx
from sqlalchemy import create_engine, text
from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication", type=Path, default=ROOT / "evidence/openstax/publication.json"
    )
    args = parser.parse_args()
    target = ROOT / "evidence/openstax/real-corpus-chat.json"
    if target.exists():
        raise RuntimeError("Preserve the previous run before recording a new attempt")
    expected_release = json.loads(args.publication.read_text("utf-8"))["release_id"]
    engine = create_engine(Settings().database_url)
    result = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": "Real HTTP, durable worker, PostgreSQL, E5 embeddings and official OpenStax passages; deterministic mock answering only. No scientific effectiveness claim.",
        "release_id": expected_release,
        "turns": [],
        "checks": {},
    }

    def save():
        target.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8"
        )

    client = httpx.Client(base_url="http://127.0.0.1:8000/api/v1", timeout=120)

    def call(method, path, **kwargs):
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()["data"]

    def login():
        token = call(
            "POST", "/auth/login", json={"email": "student2@example.com", "password": "Passw0rd!"}
        )["access_token"]
        client.headers["Authorization"] = "Bearer " + token

    def finish(receipt):
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            job = call("GET", "/jobs/" + receipt["job_id"])
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        assert job["state"] == "succeeded", job
        answer = call("GET", "/answers/" + job["answer_id"])
        assert answer["model_mode"] == "mock" and answer["response_schema"] == "chat_response_v1"
        for cid in answer["response"]["citations"]:
            ev = call("GET", f"/answers/{answer['id']}/evidence/{cid}")
            assert hashlib.sha256(ev["text"].encode()).hexdigest() == ev["text_hash"]
            assert ev["source_url"].startswith("https://assets.openstax.org/")
            assert ev["pages"] and "PDF physical pages" in ev["locator"]
        with engine.connect() as db:
            row = (
                db.execute(
                    text("SELECT release_id, trace, budget FROM answer_requests WHERE id=:id"),
                    {"id": answer["request_id"]},
                )
                .mappings()
                .one()
            )
        assert row["release_id"] == expected_release
        return answer, {
            "prepared_query": row["trace"].get("prepared_query"),
            "retrieval_candidates": row["trace"].get("retrieval_candidates"),
            "budget": row["budget"],
        }

    def ask(session_id, question, expected=None, use_profile=True):
        key = str(uuid4())
        path = f"/sessions/{session_id}/messages"
        payload = {"content": question, "use_profile": use_profile}
        submitted = time.perf_counter()
        receipt = call("POST", path, json=payload, headers={"Idempotency-Key": key})
        assert receipt == call("POST", path, json=payload, headers={"Idempotency-Key": key})
        answer, trace = finish(receipt)
        record = {
            "question": question,
            "receipt": receipt,
            "answer": answer,
            "trace": trace,
            "submission_through_evidence_verification_ms": round(
                (time.perf_counter() - submitted) * 1000, 3
            ),
        }
        result["turns"].append(record)
        save()
        if expected:
            assert answer["response"]["response_type"] == expected, record
        print(question + " => " + answer["response"]["response_type"], flush=True)
        return record

    try:
        login()
        session = call("POST", "/sessions", json={"title": "Official corpus HTTP verification"})
        result["session_id"] = session["id"]
        first = ask(session["id"], "What is photosynthesis?", "answer")
        follow = ask(session["id"], "Why does it need light?", "answer", use_profile=False)
        assert "photosynthesis" in follow["trace"]["prepared_query"]["standalone_query"].casefold()
        assert follow["answer"]["conversation_snapshot"]["messages"]
        simple = ask(session["id"], "Explain that more simply", "answer")
        assert any(e.get("inherited_from") for e in simple["answer"]["evidence"])
        source = ask(session["id"], "Show sources for that", "answer")
        assert source["answer"]["response"]["citations"]
        changed = ask(session["id"], "What is osmosis?")
        assert changed["trace"]["prepared_query"]["topic_relation"] == "new_topic"
        assert not any(e.get("inherited_from") for e in changed["answer"]["evidence"])
        previous = changed["answer"]
        feedback = call(
            "PUT",
            f"/answers/{previous['id']}/feedback",
            json={
                "helpful": True,
                "comment": "Local technical corpus verification; not a scientific rating.",
            },
        )
        regeneration_key = str(uuid4())
        regeneration = call(
            "POST",
            f"/answers/{previous['id']}/regenerate",
            headers={"Idempotency-Key": regeneration_key},
        )
        assert call("GET", "/answers/" + previous["id"])["response"] == previous["response"]
        replacement, trace = finish(regeneration)
        assert (
            replacement["id"] != previous["id"]
            and replacement["message_id"] == previous["message_id"]
        )
        assert call("GET", f"/answers/{previous['id']}/feedback")["id"] == feedback["id"]
        assert call("GET", "/answers/" + previous["id"])["evidence"] == previous["evidence"]
        result["regeneration"] = {
            "old_answer_id": previous["id"],
            "new_answer": replacement,
            "trace": trace,
            "old_feedback_id": feedback["id"],
            "old_response_and_evidence_preserved": True,
        }
        unrelated = call(
            "POST", "/sessions", json={"title": "Official corpus unrelated-input verification"}
        )
        for question in (
            "How do I configure Kubernetes ingress TLS certificates?",
            "Who won the 2026 Formula One world championship?",
            "What is the current exchange rate between the yen and the euro?",
        ):
            refusal = ask(unrelated["id"], question, "refusal")
            assert refusal["answer"]["response"]["refusal_reason"] == "INSUFFICIENT_EVIDENCE"
            assert not refusal["answer"]["response"]["citations"]
            assert refusal["trace"]["retrieval_candidates"]
        fresh = call(
            "POST", "/sessions", json={"title": "Official corpus ambiguity and cancellation"}
        )
        ask(fresh["id"], "Why does it do that?", "clarification")
        ask(fresh["id"], "Hello", "social")
        gas = ask(
            fresh["id"],
            "What happens to the pressure of a gas when its volume decreases at constant temperature?",
        )
        assert not gas["trace"]["prepared_query"]["needs_clarification"]
        assert gas["trace"]["prepared_query"]["topic_relation"] == "new_topic"
        receipt = call(
            "POST",
            f"/sessions/{fresh['id']}/messages",
            json={"content": "What is cellular respiration?", "use_profile": True},
            headers={"Idempotency-Key": str(uuid4())},
        )
        cancelled = call("POST", "/jobs/" + receipt["job_id"] + "/cancel")
        assert cancelled["state"] == "cancelled"
        time.sleep(0.8)
        assert call("GET", "/jobs/" + receipt["job_id"])["answer_id"] is None
        login()
        history = call("GET", f"/sessions/{session['id']}/messages")
        assert len(history["items"]) == 10
        assert history["items"][-1]["active_answer_id"] == replacement["id"]
        assert history["items"][1]["active_answer_id"] == first["answer"]["id"]
        result["checks"] = {
            "idempotency": True,
            "profile_off_keeps_history": True,
            "same_topic_evidence_reuse": True,
            "topic_change_uses_new_retrieval": True,
            "regeneration_preserves_old_answer_evidence_feedback": True,
            "unrelated_nonempty_candidates_refuse_in_mock": True,
            "ambiguity_and_social_separate": True,
            "current_question_local_referent": True,
            "cancel_fences_publication": True,
            "history_after_relogin": True,
        }
        result["status"] = "passed_real_pipeline_mock_answer_software_checks"
        result["limitations"] = (
            "Mock lexical selection can over-refuse supported questions; none of these outputs is live scientific or human acceptance evidence."
        )
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "detail": str(exc)[:2000]}
        raise
    finally:
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
        client.close()
        engine.dispose()
    print(json.dumps({"status": result["status"], "turns": len(result["turns"])}), flush=True)


if __name__ == "__main__":
    main()
