"""LLM adapter supporting online (OpenAI-compatible) and local HuggingFace models."""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import openai


class LLMResponse:
    """Structured LLM response."""

    def __init__(
        self,
        text: str,
        answer: Optional[str] = None,
        explanation: Optional[str] = None,
        confidence: Optional[float] = None,
        refusal_reason: Optional[str] = None,
    ):
        self.text = text
        self.answer = answer
        self.explanation = explanation
        self.confidence = confidence
        self.refusal_reason = refusal_reason


class LLMAdapter(ABC):
    """Base LLM adapter."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    @staticmethod
    def parse_answer(text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract answer letter and explanation from structured output."""
        answer_match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
        explanation_match = re.search(
            r"Explanation:\s*(.+?)(?:\n\n|$)", text, re.IGNORECASE | re.DOTALL
        )
        answer = answer_match.group(1).upper() if answer_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else None
        return answer, explanation


class OpenAIAdapter(LLMAdapter):
    """OpenAI-compatible API adapter."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content.strip()
        answer, explanation = self.parse_answer(text)
        return LLMResponse(
            text=text,
            answer=answer,
            explanation=explanation,
            confidence=None,
        )


class HuggingFaceAdapter(LLMAdapter):
    """Local HuggingFace text-generation adapter (placeholder)."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model_name = model
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self.device if self.device != "auto" else "auto",
            )
        return self._pipeline

    def generate(self, prompt: str) -> LLMResponse:
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        text = outputs[0]["generated_text"][len(prompt) :].strip()
        answer, explanation = self.parse_answer(text)
        return LLMResponse(text=text, answer=answer, explanation=explanation)
