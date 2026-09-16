"""Topic changes retrieve afresh; resolved continuations reuse the paired answer."""

from unittest.mock import patch
from app.modules.answering.models import AnswerRequest
from app.modules.answering import service
from .test_chat_runtime import corpus, session, submit, finish


def test_explicit_new_topic_retrieves_and_resolved_followup_reuses_that_topic(runtime):
    rt = runtime
    corpus(rt)
    chat = session(rt)
    photosynthesis = finish(rt, submit(rt, chat))
    with patch.object(service, "retrieve", wraps=service.retrieve) as retrieval:
        diffusion = finish(rt, submit(rt, chat, "Give an example of diffusion"))
        assert retrieval.call_count == 1
        assert retrieval.call_args.args[1] == "Give an example of diffusion"
        assert all(item["inherited_from"] is None for item in diffusion["evidence"])
        retrieval.reset_mock()
        sources = finish(rt, submit(rt, chat, "Show the sources for that"))
        assert retrieval.call_count == 0
        assert sources["evidence"] and all(
            item["inherited_from"]["request_id"] == diffusion["request_id"]
            for item in sources["evidence"]
        )
        assert all(
            item["inherited_from"]["request_id"] != photosynthesis["request_id"]
            for item in sources["evidence"]
        )
        with rt.db() as db:
            prepared = db.get(AnswerRequest, sources["request_id"]).trace["prepared_query"]
            assert prepared["topic_relation"] == "same_topic"
            assert "diffusion" in prepared["standalone_query"].lower()
            assert prepared["referenced_message_ids"]
        retrieval.reset_mock()
        new_sources = finish(rt, submit(rt, chat, "Show sources for photosynthesis"))
        assert retrieval.call_count == 1
        assert all(item["inherited_from"] is None for item in new_sources["evidence"])
