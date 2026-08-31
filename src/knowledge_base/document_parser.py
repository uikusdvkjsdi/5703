"""Document parsing for OpenStax textbooks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import requests


class DocumentParser(ABC):
    """Base document parser."""

    @abstractmethod
    def parse(self, source: str) -> Iterator[dict]:
        """Yield pages or sections as dicts with 'text', 'metadata', etc."""
        raise NotImplementedError


class OpenStaxParser(DocumentParser):
    """Parse OpenStax books via the public CMS API or local PDFs."""

    def __init__(self, api_url: str = "https://openstax.org/apps/cms/api/v2/pages"):
        self.api_url = api_url

    def parse(self, source: str) -> Iterator[dict]:
        """Parse an OpenStax book. Source can be a book slug or PDF path."""
        if source.endswith(".pdf"):
            yield from PDFParser().parse(source)
            return

        # Minimal example: list pages via the CMS API.
        response = requests.get(
            self.api_url, params={"type": "books.Book", "slug": source}, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            meta = {
                "title": item.get("title"),
                "slug": item.get("slug"),
                "id": item.get("id"),
            }
            # In a full implementation, fetch individual pages and extract text.
            yield {"text": item.get("title", ""), "metadata": meta}


class PDFParser(DocumentParser):
    """Parse PDF documents into pages."""

    def parse(self, source: str) -> Iterator[dict]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        doc = fitz.open(path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                yield {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                    },
                }
        doc.close()
