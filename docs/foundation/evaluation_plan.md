# Evaluation protocols and evidence limits

Three suites remain separate: authored chat scenarios, SciQ-derived stem-only open answers and SciQ MCQ diagnostics. Interactive chat has no dependency on SciQ or evaluator startup. A run freezes protocol/mode/condition, dataset revision/split, ordered IDs/seed, exact corpus/configuration/model/prompt/schema/profile/history versions, code revision, scorer/rubric versions and all scheduled items before test execution. Successful items never repeat on resume; failures and cancellations remain in the denominator.

The private evaluator adapter reads correct answers, support and distractors. OpenQA emits only IDs and the unmodified question stem. MCQ deterministically shuffles four distinct answer texts into A–D, with the correct label retained privately. Neither corpus ingestion nor query/model inputs can access reference/support fields. Evaluation jobs enter the shared answer components with empty profile/history; profile studies have their own typed workload. E0 has no retrieval and E1 remains basic R0 dense RAG with matched non-retrieval configuration.

Freeze at least twelve 3–5-turn families where appropriate: concept, why/how, pronoun, simplification, example, comparison, topic switch, ambiguous reference, correction, missing evidence, long history/summary, and profile/re-login continuity. Deterministic tests prove data flow and lifecycle; live conversations and actual reviewers are separately required for semantic quality claims.

| Measure | Definition and unavailable cases |
| --- | --- |
| OpenQA EM | NFKC, casefold and collapsed whitespace; nonempty compact short_answer equals reference; preserve signs/units/negation. Missing short_answer/error/refusal/cancel = 0 on scheduled denominator |
| Token F1 | Fixed word/number/math-symbol tokenizer; multiset overlap, precision=overlap/predicted, recall=overlap/reference, harmonic mean; missing/empty=0 |
| MCQ accuracy | Correct valid selections / all scheduled MCQ items; never reported as open-answer accuracy |
| Retrieval | Hit@k, precision=relevant/actual returned (empty=0), recall over judged positives, reciprocal first relevant rank, graded nDCG; missing qrels or zero IDCG unavailable |
| Citation validity | Current context ID/text/hash identity on applicable factual answers; absence is not 100%; support quality needs separate review |
| Conversation | Per-turn and complete-scenario behavior counts, including failed/cancelled turns; cluster analysis by scenario |
| Latency and usage | Separate preparation/retrieval/generation/total p50/p95 and n; actual counters; unknown token cost null |
| Profile study | Paired C0/C1/C2 × beginner/intermediate/advanced on identical frozen base/evidence/model; no learner-history writes |

No substring scoring on the full explanation (for example, “not oxygen” is not a correct oxygen answer). Aliases are reviewed/versioned before test results. Full-response correctness/support reviews use a frozen subset and report completeness. Profile review sheets randomize blind order, use independent 0–3 fit/clarity/prerequisites/usefulness/guidance/consistency ratings and separate correctness/grounding; missing ratings remain null.

Binary paired results report discordant counts, exact McNemar and paired bootstrap intervals. Continuous F1 uses paired differences and intervals, never McNemar. Repeated chat turns are not independent learners. R1 BM25, R2 RRF and R3 cross-encoder comparisons plus chunk/embedding/k ablations change one declared factor; unavailable rerankers produce blocked/unsupported results rather than R2 relabeling. Select settings only on validation. Source/config changes invalidate frozen compatibility.

Exports are separate chat_scenarios, sciq_openqa and sciq_mcq directories containing immutable manifests, per-item JSONL/CSV, aggregates, scheduled denominators, missingness and trace-linked failures. Teaching outputs and blind annotations remain separately attributable. Mock rows and historical smoke logs cannot justify model performance, learner improvement or human approval. Live configuration, actual approved corpus/SciQ and actual human ratings are external evidence inputs, not reasons to stop implementing the application.
