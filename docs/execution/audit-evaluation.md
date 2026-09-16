# Evaluation, experiment and teaching implementation audit

Scope: `evaluation/`, `configs/evaluation/`, evaluation unit tests, `tests/integration/test_evaluation_postgres.py`, and `backend/app/modules/experiment/{models,bridge,router}.py`. This audit distinguishes implemented software, observed local behavior and unperformed research/human work. It does not mark shared root/frontend/generation tasks complete on another owner's behalf.

## Executed verification

The final evaluator command and output are saved in `evaluation/software_verification.log`: five unit/application-adapter modules plus the migrated PostgreSQL integration module, **54 tests passed**, with two FastAPI/Starlette dependency deprecations. Python is 3.13.2. The shared integration fixture creates and drops a uniquely named database, applies actual Alembic migrations and invokes actual API/worker code. The experiment schema uses migration `a410a277e3a6`; independent study ownership uses `79f9729b3eae`, generated/applied by the root owner.

```text
python -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_runner.py tests/unit/test_evaluation_annotations.py tests/unit/test_evaluation_study.py tests/unit/test_evaluation_backend.py tests/integration/test_evaluation_postgres.py -q
```

There are 44 unit/application-adapter checks and 10 PostgreSQL checks. The latter cover E0 and E1 in both benchmark schemas, real configuration consumption, partial resume, source revocation before a call and after teaching generation, cancelled/stale teaching work, all nine paired teaching outputs and exports, and all twelve authored conversation families (40 turns). These are mock software tests with zero live provider calls. Source text and SciQ-shaped rows are explicitly authored. No historical source log is counted as this run. Ruff was unavailable, so there is no lint-pass claim.

## Task-by-task disposition

| Task | Implemented artifacts and observation | Remaining local work or deferred evidence |
| --- | --- | --- |
| DAT-09 | `datasets/sciq.py` loads JSON/JSONL, validates actual counts/identities and records revision/hash; projections split public inputs from private references | Acquisition of the actual SciQ revision/splits remains separate; authored four-row fixture is not an acquisition claim |
| QA-01 | Twelve 3–5-turn families, three source passages, private expected claims; parser/retrieval/MCQ negatives in tests; real HTTP capture completes all 40 scheduled turns | Root final 60-check mapping and detailed semantic/browser acceptance remain separately attributed |
| QA-02 | `datasets/sciq_openqa.py` preserves exact stem; MCQ candidate order is seed-stable and symmetric; private correct label remains separate | No remaining adapter implementation gap identified |
| QA-03 | Strict gold-free mode DTOs, public manifest/command rejection, projection perturbation tests, no evaluator imports from application modules; real benchmark requests lack dialogue/profile snapshots | Root container/mount verification must establish deployed isolation, beyond this code-level test |
| QA-04 | `annotations/qrels.py` creates/version-checks R0/manual source pools with processing/span/hash identity and nullable judgements | Real relevance/source-coverage judgements require human input; no labels invented |
| QA-05 | Conservative EM, multiset F1, MCQ equality, retrieval ranks/denominators, citation identity and null-safe aggregates; hand examples tested | Full-response semantic review remains separate from lexical proxies |
| QA-06 | `runner.py`, protocol factory and database bridge freeze all items, persist receipts/outcomes/scores, reconcile uncertainty, resume pending work, reject changed environment and export separate modes | No remaining benchmark runner implementation gap identified; CLI/Compose deployment rehearsal is root-owned |
| QA-07 | Actual shared OpenQA/MCQ baseline execution paths, E0 no-retriever instrumentation and E1 real configured R0 on authored sources | Live baselines are user-deferred; actual approved corpus/SciQ/provider setup and validation-selected runs have not been measured |
| QA-08 | `metrics/paired.py` exact binary discordants, paired bootstrap, continuous paired intervals, clustered scenario intervals; semantic ratings kept separate | Actual scientific comparison awaits the real paired outcomes/reviews |
| QA-09 | `analysis/ablations.py` checks one declared factor, matched item/config identities, validation-only choice and compatible qrels; E1 cannot be relabelled R1–R3 | No measured improvement claim; root retrieval implementations and later configured real study provide execution evidence |
| QA-10 | Nine matched study variants, random blinded packages, canonical 0–3 gates/dimensions, independent reviewer parsing, missingness/disagreement and paired-question analysis | Actual independent ratings are human-deferred; blank templates are not reviews |
| QA-11 | Evaluator-specific malformed inputs, gold/hash injection, source changes, invalid outcomes, cancellation, uncertain receipt and stale teaching budget/publication tests | General upload/security/provider/DB-failure tests belong to the root/generation integration suites |
| QA-12 | `conversations/http_backend.py` drives normal sessions/profile/re-login/messages/jobs/answers through actual API; PostgreSQL suite runs all twelve families plus evaluation/study journeys | Full J1–J6 browser interactions, latest-revision UX and responsive checks are root/frontend acceptance |
| QA-13 | Separate benchmark phase p50/p95 and usage availability; conversation capture reports its own server timing distributions/missingness and single-user concurrency; teaching job result metadata preserves actual usage/latency | Mock software timing is not live product performance; root deployment measurement and later live workload remain distinct |
| QA-14 | This scoped audit maps every applicable task/check and records actual commands/logs and limitations | Root owns the final 108-task/60-check ledger reconciliation and packaging evidence |
| PER-06 | `personalisation.study.TeachingStudyService` is reused by file and database orchestration; all nine variants share base/evidence/model, independent budgets and records; failures preserve base answer | Live matched outputs remain deferred |
| PER-08 | `annotations/blind.py` reuses the canonical rubric; opaque blind order, missing ratings, paired aggregation and disagreement are tested | Independent correctness/grounding/quality ratings remain deferred |
| BE-11 | Real experiment/study models, named-FK migration, admin lifecycle, hash registration, shared answer Job submission, independent teaching Jobs, strict scorer publication, JSON/JSONL/CSV exports | Root completes deployment/OpenAPI export; evaluator labels are never database model columns |
| CHAT-09 | Stem-only execution uses ChatResponseV1/shared generation with empty benchmark context; compact-answer scoring/versioning and preselected full-response review exports implemented | Actual OpenQA lexical/review results remain deferred; not reported as MCQ accuracy |

Shared interfaces: BE-07 uses root/generation worker claim, cancellation and recovery hooks with `execute_teaching`/`interrupt_teaching`; BE-08 uses the same answer generator/retriever/persistence, with no second answer engine; BE-12 readiness is root-owned and checks actual frozen environment/pending inputs. GEN-09 supplies one-pass product profile behavior, while this study never writes learner messages. RET-07/related retrieval tasks supply actual R1–R3 implementations; the evaluator selects and checks their declared conditions. FE-11 consumes the real protocol/count/status results and `can_start`/`can_freeze` flags. INT-06 and CHAT-12 receive the actual 40-turn API rehearsal and runnable HTTP capture; broader demo/package acceptance remains root-owned.

## Applicable acceptance and handover checks

“Covered” below means the stated software portion was directly exercised; it is not a blanket research or browser acceptance assertion.

| IDs | Scoped evidence / disposition |
| --- | --- |
| AC-07 | Covered in PostgreSQL for OpenQA and MCQ: E0 succeeds with retrieval instrumented to fail if called, empty release/evidence/history/profile and one mock provider call |
| AC-12 | Covered evaluator receipt reuse and nonduplication; repeated study start creates exactly nine independent Jobs; shared chat concurrency remains root-owned |
| AC-13 | Covered uncertain submission reconciliation and stale charged teaching-job recovery; obsolete token cannot publish and budget persists |
| AC-14 | Covered administrator-only study endpoint and source revocation/publication guard; complete cross-user resource ownership is root-owned |
| AC-17, AC-46 | Covered deterministic private/public adapters and actual stem-only/MCQ shared execution without gold; actual SciQ acquisition is not claimed |
| AC-18 | Covered all scheduled denominators, errors/refusals/cancellations, distinct protocol exports, hand metrics and provisional indicators |
| AC-19 | Covered immutable freezes, explicit effective model hashes/values, no silent prompt override, controlled-factor validation, missing qrels/ratings, preserved entered reviews and mock evidence labels |
| AC-22 | Covered actual shared application worker/generator/persistence for benchmark and interactive API capture; distinct schemas remain explicit |
| AC-23 | Covered public DTO/manifest rejection, no labels in application tables/requests and private evaluator files; Docker mount/store-credential isolation belongs to root deployment checks |
| AC-25 | Covered per-study persistent call/time budgets and attempt records, cancellation/stale work retain charges; provider retry-layer limits belong to generation tests |
| AC-27 | Covered revocation before benchmark generation (zero calls) and after teaching generation (charged call, no publication, environment_changed) |
| AC-29 | Covered separate nine-condition study outcomes and immutable original chat/base answers after failure/cancellation; one-pass product profile spy tests belong to generation |
| AC-30 | Covered real profile-off request retains the same session history; benchmark history/profile both absent |
| AC-36 | Covered canonical public OpenQA/MCQ commands; actual chat capture sends only content/use_profile; broader interactive unknown-field API checks are root-owned |
| AC-37 | Product HTTP capture operates through application APIs with authored corpus; product has no evaluator import dependency. Root owns actual stopped-service/mount-removal readiness proof |
| AC-38, AC-39, AC-40, AC-41, AC-42, AC-43 | Twelve families capture actual contextual, social, correction, source-reuse, long-history and re-login/new-session behavior. All 40 turns complete; profile/history/new-session boundaries asserted. Detailed trace/semantic assertions are in root/generation suites, not inferred from completion alone |
| AC-47 | Covered non-substring compact scoring, negation, signs, units, empty compact answers, paraphrase limitations and separate full-response/MCQ outputs |
| AC-01, AC-03, AC-08, AC-09, AC-10, AC-11, AC-15, AC-16 | Shared startup/upload/grounding/parser/provider/profile/browser checks. This scope contributes authored capture, DTO negatives and independent-study isolation; full acceptance belongs to root/generation/frontend evidence |
| AC-20, AC-21, AC-24, AC-33, AC-34, AC-35, AC-44, AC-45, AC-48 | Backup/packaging, global task DAG, tail retry, cleanup, adapter/embedding replacement, composer/latest-revision UX and legacy migration remain root/generation/frontend responsibilities; no pass is claimed from evaluator unit tests |
| HC-03 | Deterministic adapter perturbation ensures gold/support changes cannot affect public inputs; generation owner verifies full current prompt hook removal |
| HC-08, HC-09, HC-10, HC-11 | Actual provider retry/repair/prompt ordering/configuration availability regressions remain generation/root-owned; evaluator persists their terminal outcomes without redefining failures as refusals |
| HC-12 | New protocol/run directories preserve historical week5 paths; terminal results do not repeat on resume. Original authored20/two-mock/40-failure artifact preservation is separately covered by foundation/generation manifest audit |

## Remaining work distinguished by owner and evidence class

No known unimplemented requirement remains in the owned benchmark/teaching adapters, persistence, scoring or annotation tooling after the recorded local checks. Remaining local project acceptance is root-owned: final OpenAPI export/client verification, container isolation and readiness checks, clean CLI/Compose rehearsal, backup/restore, browser J1–J6 and full ledger reconciliation. This statement does not convert those pending checks into passes.

User-deferred live work: configure the actual provider/approved corpus and official SciQ revision; perform validation-selected E0/E1/variant runs with frozen manifests; measure actual paired model quality/performance. Human-deferred work: inspect source coverage/qrels, judge full-response correctness/support and supply independent blind C0–C2 ratings. Those results remain unavailable, not zero, and no research or student-learning claim is made.
