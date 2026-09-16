"""Public frozen experiments and outcomes; evaluator references have no columns."""

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import AuditMixin, Base


class ExperimentRun(Base, AuditMixin):
    __tablename__ = "experiment_runs"
    __table_args__ = (
        CheckConstraint(
            "state in ('draft','frozen','running','completed','cancelled','environment_changed','failed')",
            name="ck_experiment_run_state",
        ),
        CheckConstraint("scheduled_count > 0", name="ck_experiment_scheduled_positive"),
    )
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    protocol_id: Mapped[str] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(40))
    condition: Mapped[str] = mapped_column(String(20))
    state: Mapped[str] = mapped_column(String(40), default="draft", index=True)
    scheduled_count: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict] = mapped_column(JSON)
    manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_mode: Mapped[str] = mapped_column(String(20))
    configuration_id: Mapped[str | None] = mapped_column(
        ForeignKey("configurations.id"), nullable=True
    )
    environment_change: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ExperimentItem(Base, AuditMixin):
    __tablename__ = "experiment_items"
    __table_args__ = (
        UniqueConstraint("run_id", "item_id", name="uq_experiment_run_item"),
        UniqueConstraint("run_id", "ordinal", name="uq_experiment_run_ordinal"),
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    item_id: Mapped[str] = mapped_column(String(160))
    ordinal: Mapped[int] = mapped_column(Integer)
    question_id: Mapped[str | None] = mapped_column(String(250), nullable=True)
    command_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    command: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("answer_requests.id"), nullable=True, unique=True
    )
    state: Mapped[str] = mapped_column(String(40), default="pending")
    scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scoring_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class TeachingStudyRecord(Base, AuditMixin):
    __tablename__ = "teaching_study_records"
    __table_args__ = (
        UniqueConstraint("run_id", "item_id", name="uq_teaching_run_item"),
        CheckConstraint("condition in ('C0','C1','C2')", name="ck_teaching_condition"),
        CheckConstraint(
            "target_level in ('beginner','intermediate','advanced')", name="ck_teaching_level"
        ),
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    item_id: Mapped[str] = mapped_column(String(160))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    base_answer_id: Mapped[str] = mapped_column(ForeignKey("answers.id"))
    condition: Mapped[str] = mapped_column(String(10))
    target_level: Mapped[str] = mapped_column(String(30))
    frozen_inputs: Mapped[dict] = mapped_column(JSON)
    input_hash: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(30), default="pending")
    response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    budget: Mapped[dict] = mapped_column(JSON, default=dict)
