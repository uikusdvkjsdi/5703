# Verification and acceptance plan

The canonical top-level register is `../execution/acceptance.json`: exactly AC-01–AC-48 and HC-01–HC-12. Each check retains the original v5 expected outcome and task references. `../execution/tasks.json` contains the complete written task-level acceptance; a broad scenario cannot replace narrower task obligations. Responsive UI-01–UI-12 are subchecks in `../execution/ui_acceptance.json`, not additional top-level tasks or scenarios.

## Outcome rules

`PASS` means the entire named acceptance was executed against the current implementation and the command, exit code, environment and evidence are stored. `FAIL` means actual evidence contradicts the expected behaviour. `NOT_RUN` means no current execution exists. `PARTIAL` means only named portions were observed. `BLOCKED_EXTERNAL` identifies an exact missing asset, credential, tool/runtime or human input and does not imply independent checks are blocked. Historical logs never change a v5 check to PASS.

Implementation, technical verification, live research and human review have independent fields. Mock software checks do not establish semantic correctness or actual OpenStax quality. A fixture/schema check is not PostgreSQL transaction evidence. A screenshot is not persistence evidence. Skipped tests and missing source pages remain unverified.

## Test layers and evidence

| Layer | Fixtures and assertions | Acceptance coverage |
| --- | --- | --- |
| Foundation | All IDs/dependencies/owners/reporting links; unique registries; canonical schema/fixture parse; active docs/routes are chat-first | AC-21/48, HC-01 |
| Unit | Cleaning/spans/stable chunks; vector/cosine known ranks; BM25/RRF hand calculations; strict JSON; profile compiler; EM/F1 and denominators | AC-03–10/17–19/28/30/34/35/47, HC-04–07 |
| Contract | Pydantic/JSON Schema/OpenAPI/TypeScript equivalence; mode dispatch; actual role-separated prompt input; provider-error metadata; no gold leakage | AC-07/10/11/22/23/25/36/46, HC-02–11 |
| PostgreSQL integration | Fresh/legacy migrations, repeated seed, ownership, profile conflicts, immutable snapshots, idempotency, worker interruption/cancel, publication transactions and release rollback | AC-02/05/12–15/24–28/31–33/43/45/48 |
| Browser + real API/worker | J1–J6, answer/profile/evidence IDs and saved DB state, refresh/re-login, feedback review, no-SciQ demo, actual stop/retry/regeneration | AC-08/16/29–32/37–45 |
| Responsive browser | Width/boundary matrix, element bounds and reachable controls, local scrolling, resize state/no extra POST, focus/drawers and screenshots | UI-01–12; AC-16/37/43–45 |
| Local recovery | Isolated fresh build, backup/restore to a different target, source/evidence hashes, active release, orphan dry run and stuck-job repair | AC-01/20/33 |
| Replacement drills | Two meaningfully different local LLM adapters, PDF/TXT parsers, R0/R1, changed embedding config/release; unchanged caller contracts | AC-34/35, HC-02/08/11 |
| Separate live research | Actual pinned model/corpus/splits/runs, qrels, independent ratings, paired analyses and measured performance | AC-18/19/27/47; QA-07–10/13, RET-10/11, PER-06–09 |

Every runnable verifier must fail on a deliberate broken contract in its covered area. Avoid self-referential tests that merely restate code constants. Test request input at a spy model adapter and durable output at the database boundary where those behaviours matter.

## Required conversation families

Provide at least 12 authored families with matching source passages: (1) first factual question plus dependent follow-up; (2) simpler rephrase; (3) example request; (4) comparison; (5) missing referent then clarification; (6) greeting/social turn; (7) topic change; (8) correction of an earlier premise; (9) long-history summary and cutoff; (10) new session/profile continuity; (11) source deactivation/reuse; (12) cancel/retry/regenerate with refresh/re-login. Include profile off while continuing a dependent follow-up. Test exact included IDs, query meaning, actual provider messages, saved active revision and evidence identity. Deterministic routing does not constitute live semantic review.

## Required negative cases

Strict parsing rejects duplicate JSON keys, Markdown fences, two objects, trailing garbage, NaN/Infinity, wrong null/types and extra fields. Refusal invariants include null confidence. Provider fixtures cover empty choices, missing content, truncation, 401, 429/Retry-After, 5xx, timeout and repeatedly invalid output. At most one format repair shares persisted call/time totals with all other attempts. Cancellation/stale execution tokens cannot publish.

Mode fixtures mutate gold, evidence_status and reference strings without changing the gold-free model request. Chat has no required options; MCQ has exactly A–D with normalized duplicate rejection and exact selected text. Evaluation runtime mounts do not expose references to the API/answer worker. A missing reranker must produce explicit unavailable status instead of successful R3 fallback.

## Responsive matrix

Widths: 320, 375, 390, 640, 768, 1024, 1280, 1440, 1920 and 2560 CSS px. Boundary cases: 639/640/641 and 1023/1024/1025. Short landscape: 844×390. Resize the same loaded chat 1440→900→390→768→1440, preserving a long draft, selected session, active job/revision and no duplicate answer POST. Exercise long response/source/session titles, URLs/unbroken tokens, lists, code, tables and existing formulas; inspect local scroll reachability as well as shell width.

Check actual browser/text enlargement 125%, 150%, 200%, and equivalent 320 CSS-pixel reflow. A CSS transform/deviceScaleFactor is not actual browser zoom. Physical soft-keyboard/orientation checks require an available real browser/device and otherwise remain explicitly unavailable. Dialog Escape/Close and focus return, IME Enter, touch targets, low-height composer, Jump to latest and older-message scroll position are required.

## Evidence record format

Each execution record contains command, working-directory relation, start/end time, exit code, runtime/dependency/configuration/corpus identity, model mode, tests/assertions covered, artifact paths and exact failed/skipped/blocked cases. Append evidence rather than overwriting historical `results/week5/`. Release reporting must link every task and check to current results or a precise remaining gap.
