"""Data contracts for the independent Evaluation / QA prototype."""

from dataclasses import asdict, dataclass, field
from typing import Optional


VALID_LABELS = {"A", "B", "C", "D"}


@dataclass(frozen=True)
class GoldEvidence:
    """Reference evidence used to evaluate retrieval quality."""

    doc_id: str
    chunk_id: str
    page: Optional[int] = None


@dataclass(frozen=True)
class EvaluationExample:
    """A single fixed evaluation question."""

    question_id: str
    question: str
    choices: dict[str, str]

    gold_answer: str
    gold_answer_text: str

    gold_evidence: list[GoldEvidence] = field(
        default_factory=list
    )

    annotation_status: str = "pending"

    def __post_init__(self) -> None:
        if set(self.choices) != VALID_LABELS:
            raise ValueError(
                "choices must contain exactly A, B, C and D"
            )

        if self.gold_answer not in VALID_LABELS:
            raise ValueError(
                "gold_answer must be one of A, B, C or D"
            )

        if (
            self.choices[self.gold_answer]
            != self.gold_answer_text
        ):
            raise ValueError(
                "gold_answer_text must match "
                "the labelled gold choice"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Evaluation output for one system response."""

    question_id: str
    configuration: str

    gold_answer: str
    predicted_answer: Optional[str]

    answer_correct: bool

    retrieved_chunk_ids: list[str] = field(
        default_factory=list
    )

    hit_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    reciprocal_rank: Optional[float] = None

    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)
