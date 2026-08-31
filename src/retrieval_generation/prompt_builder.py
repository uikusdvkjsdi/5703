"""Prompt builder with evidence constraints."""

from typing import Optional


class PromptBuilder:
    """Build evidence-constrained prompts for the LLM."""

    def __init__(self, template: Optional[str] = None):
        self.template = template or self._default_template()

    @staticmethod
    def _default_template() -> str:
        return (
            "You are a helpful science tutor. Answer the multiple choice question below.\n"
            "Use only the retrieved context to support your answer. If the context is insufficient, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Choices:\n{choices}\n\n"
            "Provide your final answer as a single letter (A, B, C, or D) and a short explanation.\n"
            "Format: Answer: <letter>\\nExplanation: <reasoning>"
        )

    def format_choices(self, choices: list[str]) -> str:
        labels = ["A", "B", "C", "D"]
        return "\n".join(
            f"{label}. {text}" for label, text in zip(labels, choices)
        )

    def format_context(self, retrieved_docs: list[dict]) -> str:
        chunks = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            source = meta.get("source", "unknown")
            chunks.append(f"[{idx}] Source: {source}\n{text}")
        return "\n\n".join(chunks) if chunks else "No relevant context retrieved."

    def build(
        self,
        question: str,
        choices: list[str],
        retrieved_docs: list[dict],
    ) -> str:
        return self.template.format(
            context=self.format_context(retrieved_docs),
            question=question,
            choices=self.format_choices(choices),
        )
