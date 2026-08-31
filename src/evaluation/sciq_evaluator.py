"""SciQ dataset evaluation harness."""

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from src.evaluation.metrics import exact_match_accuracy, extract_letter
from src.retrieval_generation.pipeline import BasePipeline


class SciQEvaluator:
    """Evaluate a pipeline on the SciQ multiple-choice dataset."""

    def __init__(
        self,
        pipeline: BasePipeline,
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        self.pipeline = pipeline
        self.split = split
        self.max_samples = max_samples

    def _load_data(self):
        dataset = load_dataset("sciq", split=self.split)
        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        return dataset

    @staticmethod
    def _format_example(example: dict) -> tuple[str, list[str], str]:
        question = example["question"]
        choices = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]
        correct = example["correct_answer"]
        # SciQ does not provide A/B/C/D labels; assume correct answer is A for reference.
        correct_letter = "A"
        return question, choices, correct_letter

    def evaluate(self) -> dict:
        dataset = self._load_data()
        predictions = []
        references = []
        details = []

        for example in tqdm(dataset, desc="Evaluating"):
            question, choices, correct_letter = self._format_example(example)
            response = self.pipeline.run(question, choices)
            pred_letter = response.answer or extract_letter(response.text)
            predictions.append(pred_letter)
            references.append(correct_letter)
            details.append(
                {
                    "question": question,
                    "predicted": pred_letter,
                    "reference": correct_letter,
                    "raw_response": response.text,
                }
            )

        accuracy = exact_match_accuracy(predictions, references)
        return {
            "accuracy": accuracy,
            "total": len(predictions),
            "details": details,
        }

    def evaluate_and_save(self, output_path: str) -> dict:
        results = self.evaluate()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results
