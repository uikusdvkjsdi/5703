# Actual SciQ technical rehearsal

All four scheduled conditions completed without execution errors on 2026-09-08: two OpenQA refusals and two completed MCQ responses. They used the actual official SciQ source, real local E5 retrieval and PostgreSQL, with the **mock answer adapter**. Completed execution does not establish answer correctness. No formal accuracy, relevance, human rating or provider-cost result is claimed.

## Fixed source and conditions

The source is the acquired `allenai/sciq` revision `2c94ad3e1aafab77146f384e23536f97a4849815`, described in [the acquisition record](sciq_acquisition.md). Selection was declared before output inspection: exactly validation row indexes 0 and 1, preserving their original JSONL bytes and order. The independent review compared those bytes with the full 1,000-row split and inspected the privacy boundary before execution.

| Identity | Value |
| --- | --- |
| Full validation SHA-256 | `cf71bab38a36bdc7b6a81b6ebd20c0ab19d385b4abe1b833bd9b723ce90c1147` |
| Exact two-row prefix SHA-256 | `d55710f1bb1a209476d1d1674eb9d0fc6ea4e7eedd1b3fe56af2eb6856f80446` |
| Frozen plan SHA-256 | `1cb4ad3fcfe716e1688b4a715fe1f76633446164c3e5cbee386a1bc8eadfd82f` |
| Corpus release | `4f11bd70-a486-4d16-b216-78cfe499530a` |
| Corpus | Four official OpenStax PDFs, parser v5, 10,594 real vectors |
| Embedding | `intfloat/e5-small-v2`, revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`, actual CUDA inference |
| Retrieval | E1, R0, k=5, unchanged question stem |
| Protocols | `sciq_openqa` and `sciq_mcq`, both rows in each protocol |
| MCQ presentation seed | 5703 |
| Answer model | Mock; provider-reported usage is null |

Both selected rows pass strict MCQ preflight. No source options were repaired, no valid-only search was used, and no item was substituted after seeing an answer. The complete validation file still contains three native duplicate-choice MCQ rejections; all official splits retain 48 in total. The ordinary full-split MCQ runner still fails its strict whole-split preflight before freeze, including when a later item limit is requested. This separately named technical prefix is not the default formal dataset and does not change those denominators.

The existing `EvaluationRun.freeze`/`advance` and `DatabaseAnswerBackend` submitted work to the ordinary durable answer worker. Both protocol manifests were frozen before either ran. Their run IDs are `e8626e6a80eb4a3485a94aeacbe4bafd` (OpenQA) and `d213c3d50ce9496b832e731589cccca3` (MCQ). The report records actual corpus, visibility, model, prompt, schema and embedding fingerprints. An absent named application configuration ID is recorded as null; the effective environment fingerprints remain frozen.

## Observed timings and outcomes

Times below come directly from the four stored answers and evaluator events. Worker values are milliseconds; observer wall time is seconds.

| Protocol / source row | Response outcome | Preparation ms | Retrieval ms | Generation service ms | Worker total ms | Observer wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenQA / 0 | Refused | 46.328 | 4,624.836 | 309.360 | 5,008.813 | 5.374 |
| OpenQA / 1 | Refused | 73.498 | 4,397.410 | 299.533 | 4,798.639 | 9.747 |
| MCQ / 0 | Completed | 29.136 | 4,910.240 | 266.970 | 5,251.587 | 5.766 |
| MCQ / 1 | Completed | 32.774 | 4,551.180 | 269.187 | 4,897.480 | 10.317 |

`preparation_ms` excludes retrieval. Deterministic `query_preparation_ms` is zero because each benchmark uses its exact stem without conversation preparation. `retrieval_ms` includes the actual release/source integrity checks, query encoding and PostgreSQL search. `generation_wall_ms` measures the full generation service, including the configured 250 ms mock delay, prompt preparation, attempt accounting and response validation. The separate `generation_ms` sums adapter-measured computation across attempts: it was zero milliseconds in these four observations, not an assertion that mock computation is always zero. Other recorded local chat observations contain values from zero to one millisecond.

`total_ms` has scope `worker_execution_before_answer_insert_v1`: final answer insertion/commit and queue wait are outside that interval. Observer wall time runs from evaluator submit-start until the terminal outcome is observed, including queue wait and polling. Each protocol submits its second row while the first occupies the single worker, explaining why the second observer wall time exceeds its own worker execution time. The two protocols ran sequentially. Stages are independently measured and need not sum exactly to total time.

Every condition used one attempt and one consumed generation call, with zero format repairs and zero transient retries. All jobs succeeded, including the two valid refusal responses; every error field is null. All four recorded commands omit evaluator-private labels/support and have no conversation or profile snapshot linkage. The OpenQA refusals are retained as outcomes, not discarded from the denominator or counted as correct answers. Any existing evaluator lexical scores remain classified as mock software diagnostics and are not promoted to benchmark quality results.

The [derived timing summary](../evidence/sciq/technical-timing-summary.json) records p50/p95 separately for each protocol, using linear interpolation at `(n-1) × fraction`. Each protocol has only two observations; these are descriptive values for the complete scheduled subset, not estimated production tail latency.

| Milliseconds | OpenQA p50 | OpenQA p95 | MCQ p50 | MCQ p95 |
| --- | ---: | ---: | ---: | ---: |
| Preparation excluding retrieval | 59.913 | 72.140 | 30.955 | 32.592 |
| Deterministic query preparation | 0 | 0 | 0 | 0 |
| Retrieval | 4,511.123 | 4,613.465 | 4,730.710 | 4,892.287 |
| Generation service wall | 304.447 | 308.869 | 268.079 | 269.076 |
| Adapter computation | 0 | 0 | 0 | 0 |
| Worker total before insertion | 4,903.726 | 4,998.304 | 5,074.534 | 5,233.882 |
| Observer wall including queue/polling | 7,560.289 | 9,528.238 | 8,041.314 | 10,089.198 |

The final ledger review found a separate evaluator bridge defect: its older export path overwrote `preparation_ms` with the aggregate request-trace value that includes retrieval. The four private rehearsal states preserve those original exported values. The public rehearsal report and the table above read the correct stored `Answer.timing` values directly and are unaffected. Do not reuse the old private state's preparation field as the newer stage definition or silently amend historical run states. The [bridge correction evidence](../evidence/evaluation/preparation-timing-correction.json) records seven persisted-item regressions and an actual read-only projection of all four stored answers; no inference was rerun and historical file hashes remain unchanged.

## Evidence and reproduction

- [Frozen selection plan](../evidence/sciq/technical-rehearsal-plan.json).
- [Independent pre-execution boundary review](../evidence/sciq/technical-rehearsal-independent-review.json): source bytes and code boundaries, with no answer-quality review.
- [All four outcomes and actual stage timings](../evidence/sciq/technical-rehearsal.json).
- [Execution log](../evidence/sciq/technical-rehearsal.log).

The exact prefix and label-bearing run/reference files remain under `artifacts/evaluator-private/sciq/<revision>/technical_rehearsal/`. Public evidence exports identities, outcomes and runtime telemetry only. SciQ support was not added to the textbook corpus or learner chat.

In a fresh evidence workspace with the acquired immutable split, the project environment, PostgreSQL and the ordinary mock-configured worker available, the protocol is:

```text
python -m scripts.verify.sciq_rehearsal plan
python -m scripts.verify.sciq_rehearsal execute
```

The script deliberately refuses an existing plan or result; these commands must not overwrite this completed rehearsal. A further rehearsal requires its own explicitly versioned evidence locations and run identities. Set the intended `DATABASE_URL` and offline model environment before execution. Review actual environment fingerprints before comparing a future run. No live answer provider is permitted by this protocol. Four fixed source conditions demonstrate the technical path and timing instrumentation, not full-dataset performance, semantic validity, scalability or human learning outcomes.
