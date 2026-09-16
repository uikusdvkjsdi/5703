# Evaluation implementation verification

On 2026-09-08, Python 3.13.2 ran the five evaluation unit/application-adapter modules plus the PostgreSQL evaluation integration module: **54 passed, 2 dependency deprecation warnings**. The exact elapsed time and output are in `evaluation/software_verification.log`. Of these, 44 are unit/application-adapter tests and 10 use the shared disposable PostgreSQL fixture with actual Alembic migrations, API requests and worker execution. No paid or live provider calls were made.

Reproduce from the project root with the configured backend import path:

```text
python -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_runner.py tests/unit/test_evaluation_annotations.py tests/unit/test_evaluation_study.py tests/unit/test_evaluation_backend.py tests/integration/test_evaluation_postgres.py -q
```

The PostgreSQL tests use `tests/integration/conftest.py`: create a uniquely named database on the local server, apply migrations, seed development accounts, create an authored corpus and clean up the disposable database afterward. Source text comes from explicitly authored fixtures. They do not mutate the original project handovers or claim the historical tests as fresh v5 results.

| Acceptance IDs supported | Observed software evidence | Remaining scope |
| --- | --- | --- |
| AC-07 | Both E0 schemas execute with a retriever that would fail if invoked; requests have no release/history/profile/evidence | No live E0 performance measurement |
| AC-12, AC-13 | Interrupted evaluator submissions reconcile receipts; successful answers do not repeat; repeated teaching start creates exactly nine jobs; stale charged teaching work cannot publish or reissue | Broader chat concurrency/retry tests belong to application integration |
| AC-17, AC-46 | Stem-only projection is invariant to private reference/support/distractor changes; MCQ shuffle is deterministic and symmetric; actual shared worker receives no hidden choices in OpenQA | Official SciQ acquisition and revision verification are not represented by authored rows |
| AC-18, AC-47 | Hand fixtures cover conservative EM/F1, negation/signs/units, scheduled denominators, MCQ option equality, retrieval metrics, unknown cost and paired statistics | Actual full-response correctness/support reviews remain pending |
| AC-19 | Immutable manifests/commands/inputs reject changes, changed source/prompt/schema environments halt work, qrels and ratings stay null until entered, completed review entries survive re-export | No human-approved qrels/ratings or formal improvement finding |
| AC-22, AC-23, AC-36 | Real benchmark jobs and ordinary conversations share application worker/generator/persistence; benchmark requests have no chat snapshots/messages; server payloads reject gold; labels remain private evaluator files | Container mount separation is verified by the root integration workstream, not this module alone |
| AC-25, AC-29 | Nine C0/C1/C2 × three-level study jobs each consume one mock call with independent budgets/attempts; cancellation and late revocation preserve the original answer and citations; all failed/cancelled rows remain scheduled | Main product profile compilation is owned by the generation workstream |
| AC-30, AC-38–AC-43 | All twelve authored scenario families and forty turns complete through real HTTP/API/database/worker in mock mode; profile-off preserves history, re-login continues the session, a new session has no previous transcript | Completion is data-flow evidence. Detailed semantic correctness and browser behavior require their own checks |
| HC-12 | New runs export under separate protocol/run directories and resume without overwriting historical week5 outputs | Historical fixture preservation/manifest audit is recorded in the foundation/legacy evidence |

Implemented task portions cover DAT-09 acquisition adapters and QA-01 through QA-10 evaluator tooling, with evaluator-specific fault coverage for QA-11, real API capture for QA-12 and phase/missingness summaries for QA-13. The 108-task/60-check final audit, browser evidence and global task-state updates remain with the root integration workstream; this file does not mark those tasks complete.

Research and human evidence remain explicit: the supplied SciQ-shaped four-item dataset and three source passages are authored fixtures; no actual SciQ/OpenStax acquisition is claimed here. QA-07 live baselines, QA-09 a measured controlled improvement, QA-04 real relevance judgements and QA-10 independent reviewer ratings have not been performed. Mock outputs do not establish semantic accuracy, improved learning or human approval.

The two warnings originate from the installed FastAPI/Starlette test-client dependency stack. A lint invocation was not available because Ruff is not installed; no lint pass is claimed.
