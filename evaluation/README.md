# Independent evaluation tooling

This package evaluates the shared learning assistant. It contains no replacement answer engine. SciQ labels/support and review keys remain evaluator-only; the product does not import this package or need its datasets.

`sciq_openqa` is a project-defined stem-only diagnostic using ChatResponseV1. `sciq_mcq` is the separate four-choice selection diagnostic. `chat_scenarios` captures actual conversations and rates behaviour independently. `profile_study` orchestrates the shared TeachingStudyService on matched frozen base answers. These protocols never share an unexplained headline accuracy.

## Running the application-backed rehearsal

From the project root, with the real API/database/worker configured and the backend import path available:

```text
python -m evaluation.runner --config configs/evaluation/mock_openqa.json
python -m evaluation.runner --config configs/evaluation/mock_mcq.json
python -m evaluation.runner --resume RUN_ID --root evaluation/private_runs
```

The configs use `app.modules.experiment.bridge:create_backend`; the implementation belongs to the application integration. See `BACKEND_PROTOCOL.md` for the exact interface. Set the normal backend import environment used by Compose (project root plus backend on PYTHONPATH). Missing application integration is an explicit error, never a fallback to a canned answering adapter. `--once` performs one nonblocking submit/poll pass; an unfinished run exits 2 and is resumable. `--cancel` preserves all scheduled rows. Real model execution requires prior authorization plus `--allow-live`; all supplied defaults are mock-only.

Private immutable manifest/command/reference files are in `evaluation/private_runs/RUN_ID/`. References never appear in public registered manifests or answer commands. All items are preregistered before the first call. Submission receipts are persisted before polling. Resume polls existing receipts and never repeats terminal items. An interrupted submission is looked up by its stable key; no found receipt means uncertain, requiring reconciliation. Do not remove `runner.lock` without verifying its recorded process is gone.

In one-off containers, use mounted `EVALUATION_PRIVATE_ROOT` and `EVALUATION_EXPORT_ROOT` directories. The runner honors these environment variables for both new and resumed runs. Terminal scorer values may be published through the backend's strict `record_scores` hook for administrator results; labels/support still stay private.

Exports are under `evaluation/exports/PROTOCOL/RUN_ID/`: public immutable manifest, all scheduled JSONL/CSV outcomes, aggregates, missingness and a full-response review subset selected before execution. Costs remain null when unknown. Do not overwrite historical generation `results/week5/`. A mock rehearsal is software evidence only.

## Private data acquisition boundary

`datasets/sciq.py` reads provided JSON/JSONL official-shaped records, validates question IDs/four normalized-distinct candidates, and records actual split count/revision/SHA-256. Source file identity and declared expected count/hash must match when configured. Invalid references reject the freeze; no post-result item is removed from a denominator. The bundled four rows are explicitly authored software fixtures, not actual SciQ data or an approved OpenStax release.

OpenQA projects only question identity and the exact stem; changing private references/support/distractors leaves its command unchanged. MCQ sorts candidates then applies a per-question seeded shuffle, so changing the correct label among the same four candidate texts does not change their presentation. Only the private result contains the correct label. Neither adapter indexes SciQ support as textbook content.

## Metrics and comparisons

`metrics/scoring.py` freezes `conservative-v1`: NFKC, casefold and whitespace collapse, preserving negation/signs/numbers/units. EM requires equal nonempty compact answers. Token F1 uses multiset overlap of words, signed numbers and mathematical symbols. Full-explanation substring search is never used. Failed/refused/cancelled/incomplete/missing-answer items contribute zero to scheduled lexical denominators. Legitimate paraphrases may score poorly; full-response correctness/support review remains separate.

Retrieval uses relevant/actual-returned precision, judged-positive recall, first-relevant reciprocal rank and graded nDCG. Missing qrels and zero IDCG remain unavailable as applicable; unjudged results are counted visibly. Citation identifier/text-hash validity is distinct from factual support. Binary paired scores use exact McNemar discordants and paired bootstrap intervals; continuous scores use paired differences. Repeated turns are clustered by scenario and do not receive an invalid per-turn McNemar test.

`analysis/ablations.py` requires one declared factor, fixed question/source/model/prompt/scorer identities and validation-only setting selection. R1/R2/R3 cannot be relabelled E1. Changed chunks require compatible source-span judgements or reannotation. A compatible experiment design is not an executed scientific finding.

## Source review and blind ratings

```text
python -m evaluation.annotations.cli qrels-template --candidates CANDIDATES.json --output PRIVATE_QRELS.json
python -m evaluation.annotations.cli qrels-validate --input PRIVATE_QRELS.json --release RELEASE --processing PROCESSING_ID
python -m evaluation.annotations.cli blind-template --outputs STUDY_OUTPUTS.json --destination PRIVATE_REVIEW_DIR --seed 5703
python -m evaluation.annotations.cli ratings-analyze --package PRIVATE_REVIEW_DIR/private_package.json --ratings COMPLETED_RATINGS.csv --reviewers REVIEWER_A REVIEWER_B --output ANALYSIS.json
```

Qrel templates pin release, processing IDs, exact source spans and text hashes. Null grades stay present until a real reviewer/date is recorded. Changed pools need a new version; source overlap is a review aid, not automatic transferred relevance.

Blinded packages retain all C0/C1/C2 × beginner/intermediate/advanced items, including failed outputs. Base answer, evidence and model must match per question. The private key maps opaque IDs to conditions; the reviewer view omits those identities. Ratings use the canonical `personalisation.rubric` 0–3 anchors for six dimensions and correctness/groundedness gates (pass at ≥2). Empty ratings remain null, failed outputs cannot receive invented ratings, duplicate reviewer/item entries are rejected, and paired analysis aggregates independent reviewers by question. Output ratings do not measure student learning gains.

`python -m evaluation.study --config STUDY_CONFIG.json --blind-output PRIVATE_REVIEW_DIR` freezes and runs the nine-condition design through `personalisation.study.TeachingStudyService`. The config provides `questions_path`, `private_root`, optional `model_config` and seed. Each question supplies question_id/question/base_answer/evidence from real saved output. `--resume PRIVATE_STUDY_DIR` resumes pending items; uncertain paid execution never repeats automatically. These files do not write or mutate learner messages.

The application also persists this independent study in PostgreSQL: administrator `POST /teaching-studies` with saved neutral `base_answer_ids`, seed and optional configuration_id freezes all nine variants per question; start/cancel/results/export use the returned `/experiments/{id}` endpoints. Each worker job records its own budget and attempt history. Existing answers and citations remain immutable when a study fails or is cancelled. Download its JSON export and give it directly to `blind-template`. Actual independent ratings are still required; the API never fabricates them.

## Conversation fixtures and tests

`conversations/scenarios.json` contains 12 distinct 3–5-turn families. `source_requirements.json` describes matching authored corpus passages; private review claims are in `private_fixture/conversation_expectations.json`. A real browser or `ChatBackend` implementation supplies session/profile/re-login operations. The driver sends only ordinary content/use_profile and saves observed results. Completion rates are data-flow measures, with semantic review null until actual review occurs.

`python -m evaluation.conversations.http_backend --email REHEARSAL_ACCOUNT --output evaluation/exports/chat_scenarios/UNIQUE_CAPTURE` connects to the actual HTTP API and worker. Set `EVALUATION_ACCOUNT_PASSWORD` in the environment. Use a dedicated rehearsal account: the profile-continuity case updates its saved profile, and all conversations persist normally. The CLI freezes the scenario manifest before calls, saves each observed scenario, reports separate conversation timing p50/p95 and leaves unavailable phases/cost null. It is a capture tool, not an automatic replay/resume mechanism: an interrupted server job must be reconciled through its saved application state before a new capture. No credentials or private expected answers are exported.

Run the focused software suite:

```text
python -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_runner.py tests/unit/test_evaluation_annotations.py tests/unit/test_evaluation_study.py tests/unit/test_evaluation_backend.py tests/integration/test_evaluation_postgres.py -q
```

These tests exercise hand calculations, leakage invariants, frozen/resumable state, uncertainty/cancellation, source-change halts, blinded missingness and the actual shared mock teaching service. The integration module creates a disposable PostgreSQL database, applies real migrations and drives real APIs/workers, including all twelve authored conversation families. Browser acceptance, actual SciQ/OpenStax acquisition, live research and human review remain distinct evidence. See `VERIFICATION.md` for the scope of the recorded run.
