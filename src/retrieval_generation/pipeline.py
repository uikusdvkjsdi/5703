"""E0 and E1 generation pipelines."""

from abc import ABC, abstractmethod
from typing import Optional

from src.knowledge_base.embedder import DenseEmbedder
from src.knowledge_base.vector_store import BaseVectorStore
from src.retrieval_generation.llm_adapter import LLMAdapter, LLMResponse
from src.retrieval_generation.prompt_builder import PromptBuilder
from src.retrieval_generation.retriever import DenseRetriever


class BasePipeline(ABC):
    """Common interface for E0 and E1 pipelines."""

    def __init__(self, llm: LLMAdapter, prompt_builder: Optional[PromptBuilder] = None):
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()

    @abstractmethod
    def run(self, question: str, choices: list[str]) -> LLMResponse:
        raise NotImplementedError


class PureLLMPipeline(BasePipeline):
    """E0: Pure LLM baseline without retrieval."""

    def run(self, question: str, choices: list[str]) -> LLMResponse:
        prompt = self.prompt_builder.build(
            question=question,
            choices=choices,
            retrieved_docs=[],
        )
        return self.llm.generate(prompt)


class RAGPipeline(BasePipeline):
    """E1: Basic dense RAG with retrieval."""

    def __init__(
        self,
        llm: LLMAdapter,
        vector_store: BaseVectorStore,
        embedder: DenseEmbedder,
        prompt_builder: Optional[PromptBuilder] = None,
        top_k: int = 5,
    ):
        super().__init__(llm, prompt_builder)
        self.retriever = DenseRetriever(vector_store, embedder)
        self.top_k = top_k

    def run(self, question: str, choices: list[str]) -> LLMResponse:
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
        prompt = self.prompt_builder.build(
            question=question,
            choices=choices,
            retrieved_docs=retrieved_docs,
        )
        return self.llm.generate(prompt)
