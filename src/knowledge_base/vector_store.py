"""Vector store backends: FAISS and pgvector."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np


class BaseVectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self._documents: list[dict] = []
        self._index: Optional[faiss.Index] = None
        self._init_index()

    def _init_index(self) -> None:
        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.metric}")

    def add(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self._index.add(embeddings.astype(np.float32))
        self._documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self._index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[int(idx)].copy()
            doc["score"] = float(score)
            results.append(doc)
        return results

    def save(self, path: Optional[str] = None) -> None:
        if path is None:
            raise ValueError("Path is required to save FAISS index")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, path)

    def load(self, path: Optional[str] = None) -> None:
        if path is None:
            raise ValueError("Path is required to load FAISS index")
        self._index = faiss.read_index(path)


class PGVectorStore(BaseVectorStore):
    """pgvector-backed vector store (placeholder implementation)."""

    def __init__(self, dsn: str, collection_name: str, dimension: int):
        self.dsn = dsn
        self.collection_name = collection_name
        self.dimension = dimension
        raise NotImplementedError(
            "pgvector backend is not implemented in this baseline scaffold."
        )

    def add(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def save(self, path: Optional[str] = None) -> None:
        raise NotImplementedError

    def load(self, path: Optional[str] = None) -> None:
        raise NotImplementedError
