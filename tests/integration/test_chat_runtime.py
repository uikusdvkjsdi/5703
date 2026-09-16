"""Lifecycle and source invariants against the real API and PostgreSQL."""

from uuid import uuid4
import hashlib
from sqlalchemy import select, func
from app.modules.answering.models import Answer, AnswerRequest
from app.modules.answering.service import execute_answer


def call(rt, method, path, body=None, headers=None, status=200):
    response = rt.client.request(
        method, "/api/v1" + path, json=body, headers=headers or rt.headers()
    )
    assert response.status_code == status, response.text
    return response.json().get("data", response.json())


def corpus(rt):
    admin = rt.headers("admin@example.com")
    raw = b"# Photosynthesis\nPhotosynthesis captures light energy and stores chemical energy in sugars. Light supplies energy to convert carbon dioxide and water into sugars. Oxygen is released.\n\n# Diffusion\nDiffusion is the net movement of particles from higher concentration to lower concentration. Dye spreading through water is an example."
    response = rt.client.post(
        "/api/v1/documents",
        headers=admin,
        data={"title": "Authored test biology " + uuid4().hex[:6]},
        files={
            "file": ("source-" + uuid4().hex + ".txt", raw + uuid4().hex.encode(), "text/plain")
        },
    )
    assert response.status_code == 201, response.text
    doc = response.json()["data"]["document"]
    process = call(rt, "POST", "/documents/" + doc["id"] + "/process", {}, admin, 202)
    assert rt.work()
    job = call(rt, "GET", "/jobs/" + process["job_id"], headers=admin)
    assert job["state"] == "succeeded", job
    build = call(
        rt,
        "POST",
        "/corpus/releases",
        {"processing_run_ids": [process["processing_id"]]},
        admin,
        202,
    )
    assert rt.work()
    call(rt, "POST", "/corpus/releases/" + build["release_id"] + "/activate", {}, admin)
    return doc, build


def session(rt):
    return call(rt, "POST", "/sessions", {"title": "Lifecycle " + uuid4().hex[:8]})


def submit(rt, s, text="What is photosynthesis?", key=None, profile=True):
    headers = {**rt.headers(), "Idempotency-Key": key or str(uuid4())}
    return call(
        rt,
        "POST",
        f"/sessions/{s['id']}/messages",
        {"content": text, "use_profile": profile},
        headers,
        202,
    )


def finish(rt, receipt):
    assert rt.work()
    job = call(rt, "GET", "/jobs/" + receipt["job_id"])
    assert job["state"] == "succeeded", job
    return call(rt, "GET", "/answers/" + job["answer_id"])


def test_atomic_chat_context_and_current_evidence(runtime):
    rt = runtime
    corpus(rt)
    s = session(rt)
    first = finish(rt, submit(rt, s))
    second = finish(rt, submit(rt, s, "Why does it need light?"))
    assert first["response"]["response_type"] == "answer"
    assert len(second["conversation_snapshot"]["messages"]) == 2
    with rt.db() as db:
        req = db.get(AnswerRequest, second["request_id"])
        assert "photosynthesis" in req.trace["prepared_query"]["standalone_query"].casefold()
        assert req.budget["consumed_calls"] == 1
        assert len(req.trace["model_messages"]) >= 4
        assert (
            db.scalar(select(func.count()).select_from(Answer).where(Answer.request_id == req.id))
            == 1
        )
    for cid in second["response"]["citations"]:
        ev = call(rt, "GET", f"/answers/{second['id']}/evidence/{cid}")
        assert hashlib.sha256(ev["text"].encode()).hexdigest() == ev["text_hash"]
    page = call(rt, "GET", f"/sessions/{s['id']}/messages?limit=2")
    assert len(page["items"]) == 2 and page["next_after_sequence"] == 2


def test_idempotency_busy_and_forbidden_fields(runtime):
    rt = runtime
    s = session(rt)
    key = str(uuid4())
    receipt = submit(rt, s, "Hello", key)
    assert submit(rt, s, "Hello", key) == receipt
    h = {**rt.headers(), "Idempotency-Key": key}
    call(
        rt,
        "POST",
        f"/sessions/{s['id']}/messages",
        {"content": "Different", "use_profile": True},
        h,
        409,
    )
    h["Idempotency-Key"] = str(uuid4())
    call(
        rt,
        "POST",
        f"/sessions/{s['id']}/messages",
        {"content": "Next", "use_profile": True},
        h,
        409,
    )
    call(rt, "POST", f"/sessions/{s['id']}/messages", {"content": "Hi", "gold": "secret"}, h, 422)
    answer = finish(rt, receipt)
    assert answer["response"]["response_type"] == "social" and not answer["evidence"]


def test_profile_snapshot_conflict_and_profile_off_preserves_history(runtime):
    rt = runtime
    corpus(rt)
    s = session(rt)
    profile = call(rt, "GET", "/profiles/me")
    receipt = submit(rt, s)
    call(rt, "PUT", "/profiles/me", {**profile, "level": "advanced"})
    call(rt, "PUT", "/profiles/me", {**profile, "level": "beginner"}, status=409)
    answer = finish(rt, receipt)
    assert answer["profile_snapshot"]["source_version"] == profile["version"]
    second = finish(rt, submit(rt, s, "Explain that more simply", profile=False))
    assert second["conversation_snapshot"]["messages"]
    assert second["profile_snapshot"]["profile"] is None
    assert second["profile_snapshot"]["turn_override"]["reason"] == "explicit_simplification"
    assert "precise terminology" not in second["profile_snapshot"]["policy"]
    assert call(rt, "GET", "/profiles/me")["level"] == "advanced"


def test_cancel_retry_and_stale_publication(runtime):
    rt = runtime
    s = session(rt)
    receipt = submit(rt, s, "Hello")
    call(rt, "POST", "/jobs/" + receipt["job_id"] + "/cancel", {})
    old = call(rt, "GET", "/jobs/" + receipt["job_id"])
    assert old["state"] == "cancelled" and old["can_retry"]
    headers = {**rt.headers(), "Idempotency-Key": str(uuid4())}
    retried = call(
        rt, "POST", "/answer-requests/" + receipt["request_id"] + "/retry", {}, headers, 202
    )
    assert retried["job_id"] != receipt["job_id"] and retried["request_id"] == receipt["request_id"]
    assert (
        call(rt, "POST", "/answer-requests/" + receipt["request_id"] + "/retry", {}, headers, 202)
        == retried
    )
    execute_answer(rt.engine, rt.settings, receipt["job_id"], "stale-token")
    finish(rt, retried)
    assert len(call(rt, "GET", f"/sessions/{s['id']}/messages")["items"]) == 2
    assert call(rt, "GET", "/jobs/" + receipt["job_id"])["state"] == "cancelled"


def test_latest_revision_preserves_feedback_and_failed_replacement(runtime):
    rt = runtime
    doc, _ = corpus(rt)
    s = session(rt)
    a = finish(rt, submit(rt, s))
    feedback = call(
        rt,
        "PUT",
        "/answers/" + a["id"] + "/feedback",
        {"helpful": True, "comment": "Original revision"},
    )
    headers = {**rt.headers(), "Idempotency-Key": str(uuid4())}
    replacement = call(rt, "POST", "/answers/" + a["id"] + "/regenerate", {}, headers, 202)
    assert (
        call(rt, "GET", f"/sessions/{s['id']}/messages")["items"][-1]["active_answer_id"] == a["id"]
    )
    b = finish(rt, replacement)
    assert b["id"] != a["id"] and b["message_id"] == a["message_id"]
    assert call(rt, "GET", "/answers/" + a["id"] + "/feedback")["id"] == feedback["id"]
    headers["Idempotency-Key"] = str(uuid4())
    call(rt, "POST", "/answers/" + a["id"] + "/regenerate", {}, headers, 409)
    call(rt, "POST", "/documents/" + doc["id"] + "/revoke", {}, rt.headers("admin@example.com"))
    replacement = call(rt, "POST", "/answers/" + b["id"] + "/regenerate", {}, headers, 202)
    assert rt.work()
    assert call(rt, "GET", "/jobs/" + replacement["job_id"])["state"] == "failed"
    assert (
        call(rt, "GET", f"/sessions/{s['id']}/messages")["items"][-1]["active_answer_id"] == b["id"]
    )
    call(rt, "GET", f"/answers/{a['id']}/evidence/{a['response']['citations'][0]}", status=410)


def test_ownership_archive_and_new_session_isolation(runtime):
    rt = runtime
    s = session(rt)
    receipt = submit(rt, s, "Hello")
    call(
        rt,
        "GET",
        f"/sessions/{s['id']}/messages",
        headers=rt.headers("student2@example.com"),
        status=404,
    )
    call(rt, "GET", "/documents", status=403)
    call(rt, "POST", f"/sessions/{s['id']}/archive", {})
    assert call(rt, "GET", "/jobs/" + receipt["job_id"])["state"] == "cancelled"
    fresh = session(rt)
    answer = finish(rt, submit(rt, fresh, "Why does it do that?"))
    assert (
        answer["response"]["response_type"] == "clarification"
        and not answer["conversation_snapshot"]["messages"]
    )
