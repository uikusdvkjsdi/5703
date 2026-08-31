"""Structure-aware text chunking."""

import re
from typing import Iterator


class StructureAwareChunker:
    """Split documents into chunks while respecting headings and size limits."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        respect_headings: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_headings = respect_headings

    def _split_by_headings(self, text: str) -> list[str]:
        if not self.respect_headings:
            return [text]
        # Split on lines that look like markdown headings or all-caps headings.
        pattern = re.compile(r"(?:\n|^)(#{1,6}\s+.+|\d+(?:\.\d+)*\s+[A-Z][A-Za-z\s]+)")
        parts = pattern.split(text)
        sections = []
        current = ""
        for part in parts:
            if pattern.match(part):
                if current.strip():
                    sections.append(current.strip())
                current = part
            else:
                current += part
        if current.strip():
            sections.append(current.strip())
        return sections or [text]

    def _chunk_text(self, text: str) -> Iterator[str]:
        words = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                yield chunk

    def chunk(self, document: dict) -> Iterator[dict]:
        """Yield chunks from a parsed document."""
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        for section_idx, section in enumerate(self._split_by_headings(text)):
            for chunk_idx, chunk_text in enumerate(self._chunk_text(section)):
                chunk_meta = {
                    **metadata,
                    "section_idx": section_idx,
                    "chunk_idx": chunk_idx,
                }
                yield {"text": chunk_text, "metadata": chunk_meta}
