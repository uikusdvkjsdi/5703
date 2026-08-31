"""Run retrieval and generation on a SciQ-style question."""

import argparse

import yaml

from src.knowledge_base.embedder import DenseEmbedder
from src.knowledge_base.vector_store import FAISSVectorStore
from src.retrieval_generation.llm_adapter import LLMAdapter, OpenAIAdapter
from src.retrieval_generation.pipeline import PureLLMPipeline, RAGPipeline
from src.retrieval_generation.prompt_builder import PromptBuilder


def load_llm(llm_config: dict) -> LLMAdapter:
    provider = llm_config["provider"]
    if provider == "openai":
        return OpenAIAdapter(
            model=llm_config["model"],
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 256),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def main(config_path: str, mode: str, question: str, choices: list[str]) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rg_config = config["retrieval_generation"]
    llm = load_llm(rg_config["llm"])
    prompt_builder = PromptBuilder(template=rg_config["prompt"]["template"])

    if mode == "e0":
        pipeline = PureLLMPipeline(llm=llm, prompt_builder=prompt_builder)
    elif mode == "e1":
        kb_config = config["knowledge_base"]
        embedder = DenseEmbedder(**kb_config["embedding"])
        vector_store = FAISSVectorStore(
            dimension=embedder.dimension,
            metric=kb_config["vector_store"]["faiss"]["metric"],
        )
        vector_store.load(kb_config["vector_store"]["faiss"]["index_path"])
        pipeline = RAGPipeline(
            llm=llm,
            vector_store=vector_store,
            embedder=embedder,
            prompt_builder=prompt_builder,
            top_k=rg_config.get("top_k", 5),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'e0' or 'e1'.")

    response = pipeline.run(question, choices)
    print("Mode:", mode)
    print("Answer:", response.answer)
    print("Explanation:", response.explanation)
    print("Raw response:\n", response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval and generation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", default="e1", choices=["e0", "e1"], help="Baseline mode")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument(
        "--choices",
        required=True,
        nargs=4,
        help="Four answer choices",
    )
    args = parser.parse_args()
    main(args.config, args.mode, args.question, args.choices)
