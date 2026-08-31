"""Run SciQ evaluation for E0 and E1 baselines."""

import argparse
import json
from pathlib import Path

import yaml

from src.evaluation.sciq_evaluator import SciQEvaluator
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


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rg_config = config["retrieval_generation"]
    eval_config = config["evaluation"]
    llm = load_llm(rg_config["llm"])
    prompt_builder = PromptBuilder(template=rg_config["prompt"]["template"])

    results = {}

    for mode in eval_config["baselines"]:
        print(f"\n=== Running baseline {mode.upper()} ===")
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
            raise ValueError(f"Unknown baseline: {mode}")

        evaluator = SciQEvaluator(
            pipeline=pipeline,
            split=eval_config["split"],
            max_samples=eval_config.get("max_samples"),
        )
        mode_results = evaluator.evaluate_and_save(
            str(Path(eval_config["output_path"]).with_suffix(f".{mode}.json"))
        )
        results[mode] = mode_results
        print(f"{mode.upper()} accuracy: {mode_results['accuracy']:.4f}")

    summary_path = Path(eval_config["output_path"]).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate E0 and E1 baselines on SciQ")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
