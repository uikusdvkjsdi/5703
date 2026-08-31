"""Tests for the prompt builder."""

from src.retrieval_generation.prompt_builder import PromptBuilder


def test_format_choices():
    builder = PromptBuilder()
    choices = ["one", "two", "three", "four"]
    formatted = builder.format_choices(choices)
    assert "A. one" in formatted
    assert "D. four" in formatted


def test_format_context():
    builder = PromptBuilder()
    docs = [{"text": "sample context", "metadata": {"source": "test"}}]
    context = builder.format_context(docs)
    assert "sample context" in context
    assert "test" in context
