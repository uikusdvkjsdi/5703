"""Dense retriever for top-k context retrieval."""

import numpy as np

from src.knowledge_base.embedder import DenseEmbedder
from src.knowledge_base.vector_store import BaseVectorStore


class DenseRetriever:
    """Retrieve top-k relevant chunks for a query."""

    def __init__(self, vector_store: BaseVectorStore, embedder: DenseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.encode(query)
        return self.vector_store.search(query_embedding, top_k=top_k)
