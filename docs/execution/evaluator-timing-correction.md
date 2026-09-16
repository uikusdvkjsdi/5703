# Evaluator preparation-timing correction

The evaluator item projection overwrote the saved answer's measured `preparation_ms` with `AnswerRequest.trace.preparation_ms`. The request trace is the older aggregate of preparation and retrieval. Combining that aggregate with the answer's separate retrieval stage counted retrieval twice while retaining the newer `worker_execution_before_answer_insert_v1` label.

`backend/app/modules/experiment/bridge.py` now preserves the complete saved answer timing dictionary whenever values exist. Measured zero remains zero. A missing value under an explicit timing scope remains missing; an aggregate is not substituted under a measured-stage label. Only an unscoped legacy answer missing preparation falls back to the historical request aggregate, with explicit `preparation_timing_scope=legacy_preparation_including_retrieval` and `preparation_timing_source=request_trace.preparation_ms` metadata. Missing legacy trace data remains null. Neither saved timing nor request trace is modified.

The four original private SciQ rehearsal outcomes remain unchanged. Their stored preparation values are historical export defects; `evidence/sciq/technical-rehearsal.json` already contains the correct immutable answer timings. The correction report reads the same saved requests/answers through the corrected projection without provider or embedding inference:

| Protocol/item | Historical exported preparation (ms) | Correct saved answer and current projection (ms) |
| --- | ---: | ---: |
| OpenQA / item_000000 | 4671.163 | 46.328 |
| OpenQA / item_000001 | 4470.908 | 73.498 |
| MCQ / item_000000 | 4939.376 | 29.136 |
| MCQ / item_000001 | 4583.954 | 32.774 |

Evidence: [preparation-timing-correction.json](../../evidence/evaluation/preparation-timing-correction.json). The actual PostgreSQL read used an enforced read-only transaction. All four corrected outcome dictionaries match the saved answer and already-correct public report. Before/after hashes confirm that the two private state files and public report were preserved. Only timing values, public identifiers and hashes are included; evaluator-private questions, answers and scores are not copied into this report.

Seven focused regressions in `tests/unit/test_evaluation_outcome_timing.py` passed using isolated in-memory SQLite persistence and the real `item_outcome`/`answer_out` path, with no provider calls. Cases cover distinct answer/trace values, measured zero, existing unscoped values, labelled legacy fallback, missing legacy data, missing scoped measurements and an unknown future scope. The checks also verify that exporting creates no dirty database objects or changes to stored timing/trace data. Ruff formatting and correctness checks passed for the two changed executable files. Pytest reported a cache-directory permission warning; the seven tests passed.

This is a correction to timing export, not a new inference run, benchmark score or scientific-quality result. Historical exported outcomes remain inspectable. The root owner controls the final aggregate gate and runtime restart; this report does not claim that an already running process automatically loaded the new bridge code.

The root subsequently restarted the API/worker after the fixed-corpus queue drained. Actual HTTP results for both existing runs matched all four saved timing dictionaries: [restarted API proof](../../evidence/evaluation/restarted-api-timing-projection.json). No answering or embedding calls were repeated.
