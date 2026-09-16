# Shared answer backend contract

The factory specified by a trusted evaluator CLI config receives `backend_options` as one dict and returns an object implementing `evaluation.bridge.AnswerBackend`. Root application factory target: `app.modules.experiment.bridge:create_backend`. This is an injected transport boundary, not another answer engine.

Methods are synchronous; provider work remains in the application's durable worker:

```python
environment() -> dict
register_run(manifest: dict) -> dict
submit(command: dict, *, run_context: dict, idempotency_key: str) -> dict
poll(receipt: dict) -> dict
lookup(idempotency_key: str) -> dict | None
```

`environment()` returns actual model_mode, configuration_id, configuration_hash, corpus_release_id, corpus_manifest_hash, model_configuration_hash, prompt_configuration_hash, response_schema_hash, embedding_configuration_hash and source_visibility_hash. The application bridge requires that complete set; a changed/missing value halts new submissions and marks environment_changed. It does not silently select a new source release. Prompts and response schemas are hashed from the actual loaded implementation.

`register_run()` receives the public manifest containing version, run_id, protocol_id, mode, condition, dataset identity/revision/split/file hash, ordered items `{item_id,question_id,command_hash}`, seed, scheduled_count, environment, configuration, scorer/rubric/code versions and manifest_hash. It contains no reference text, support, correct labels, ratings or private file paths. It must be idempotent for the same ID/hash and reject changed manifests. Return `{run_id: same_id}`. The local manifest hash excludes the manifest_hash field itself.

`submit()` receives exactly canonical OpenQACommand or MCQCommand fields. OpenQA: mode, run_id, item_id, question_id, question_text. MCQ adds options with exactly A–D. The trusted run_context is the public frozen manifest; condition/configuration are taken from that manifest, not caller-selected model fields in the question DTO. Benchmarks use no dialogue/profile/summary. The idempotency key is `evaluation:<run_id>:<item_id>`. Return a durable receipt with nonempty request_id and job_id before polling. Registration/submission must not store evaluator references in public tables.

`poll()` returns `{status: "pending"}` until terminal. A terminal outcome has status completed/refused/error/cancelled/invalid/incomplete; completed requires a `response` dict using ChatResponseV1 or MCQResponseV1. Include model_mode, request_id/answer_id, evidence/candidate IDs, context identities, timings `{preparation_ms,retrieval_ms,generation_ms,total_ms}`, usage with nullable token/cost totals, or a standard safe error `{code,message,trace_id}` when available. A valid model refusal should report refused and preserve its typed response. Never map provider failure to refusal. Live and mock status must be accurate.

`lookup()` reconciles an interrupted submit using the same key. A found receipt resumes polling; None leaves the item uncertain and does not authorize a fresh submission. Terminal items never repeat. Backend polling/transport exceptions keep the durable receipt for the next resume. A crash before a receipt is saved remains explicit; the runner never claims exactly-once external calls.

The runner first writes its immutable private manifest/commands/references and preregisters every pending item. It atomically persists submitting/submitted/terminal state. Results use scheduled denominators, retaining failures/cancellations. Private paths are only evaluator storage. No raw data, scoring module or runtime mount is required by product chat.

Optional `record_scores(run_id, item_id, scores)` receives only the frozen scorer's numeric/boolean outputs and version after the outcome is terminal. The runner retries score publication without repeating answer generation. Optional `cancel_run(run_id)` cancels all unfinished server items while preserving previous terminal outcomes. These extensions keep administrator result counts connected to separate evaluator scoring without importing the scorer or labels into the application.

For disposable evaluator containers, set `EVALUATION_PRIVATE_ROOT` and `EVALUATION_EXPORT_ROOT` to mounted directories. They override local path defaults so a one-off container does not discard resumable state.

The immutable application `Configuration.values.model` object may override model name, window_tokens, max_tokens (at most 1024), timeout_seconds (at most 60), temperature, seed, token_limit_parameter, structured_output_mode and reasoning_effort. These values are validated, included in the effective model hash and passed to the actual shared generator. Provider identity, endpoint and credential environment remain server settings. Unknown model fields and arbitrary prompt override fields are rejected. Runtime condition selects E0/no retrieval, E1/R0 or the matching R1–R3 implementation. Explicit manifest top_k takes precedence over the frozen configuration's top_k; absence of both uses the pinned release configuration.

Administrator `POST /teaching-studies` accepts only `base_answer_ids`, seed and optional configuration_id. Saved completed neutral grounded answers supply the exact base and original evidence. It preregisters all C0/C1/C2 × three levels, randomizes their queue order and writes independent teaching records before any calls. Use the returned ID with `/experiments/{id}/start`, `/cancel`, `/results` and `/export`. Each item uses `TeachingStudyService`, its own persisted budget and attempt log, and a `Job` with no AnswerRequest or learner-message writes. Start is idempotent. Source/model/prompt/schema/profile-rule changes prevent publication; stale charged calls fail explicitly without reissue. Results retain every outcome and have null human-review scores. Save an export JSON and pass it directly to `evaluation.annotations.cli blind-template`; the annotation CLI accepts either the HTTP envelope, result object or raw output list.
