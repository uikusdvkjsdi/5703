# Reproduce the evaluator-only SciQ acquisition

The actual SciQ dataset was acquired on 2026-09-08 from the research owner's [`allenai/sciq` repository](https://huggingface.co/datasets/allenai/sciq/tree/2c94ad3e1aafab77146f384e23536f97a4849815). The [original Ai2 dataset address](https://allenai.org/data/sciq) redirects there. Its fixed revision is `2c94ad3e1aafab77146f384e23536f97a4849815`, last modified 2024-01-04. The source card identifies **CC BY-NC 3.0**. Retain its attribution to Johannes Welbl, Nelson F. Liu and Matt Gardner, [Crowdsourcing Multiple Choice Science Questions (2017)](https://aclanthology.org/W17-4413/).

The acquisition contains all 13,679 original records. Source Parquet files, the original source card, metadata, derived JSONL, reference answers and anomaly details are retained under `artifacts/evaluator-private/sciq/2c94ad3e1aafab77146f384e23536f97a4849815/`. Original field values and split order are unchanged. Source support is never ingested into OpenStax, retrieval, learner conversations or provider evidence.

| Split | Original records | Empty source support | OpenQA ready | MCQ ready | Native duplicate-choice rejections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,679 | 1,198 | 11,679 | 11,643 | 36 |
| Validation | 1,000 | 113 | 1,000 | 997 | 3 |
| Test | 1,000 | 116 | 1,000 | 991 | 9 |
| Total | 13,679 | 1,427 | 13,679 | 13,631 | 48 |

Empty support is a native source property and is retained. It does not prevent stem-only questions or imply that textbook evidence is unavailable.

| Original file | Bytes | SHA-256 |
| --- | ---: | --- |
| `train-00000-of-00001.parquet` | 3,993,099 | `19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167` |
| `validation-00000-of-00001.parquet` | 338,503 | `455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a` |
| `test-00000-of-00001.parquet` | 342,808 | `3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667` |

These values match the pinned repository's LFS SHA-256 values and split counts. `evidence/sciq/acquisition.json` also records source URLs, derived-file hashes, reader version, acquisition date and source-card hash. The current snapshot is not relabelled as a historical project fixture or a previous research run.

## Reproduction

From the project root, use the optional acquisition dependency and commands:

```text
python -m pip install -r requirements-sciq.lock
python -m scripts.dev.acquire_sciq
python -m scripts.verify.sciq
```

`requirements-sciq.lock` pins PyArrow 25.0.1; the actual CPython 3.13 Windows wheel and hash are recorded in the installation report. The application and evaluator runner need no PyArrow dependency after conversion. The acquisition script accepts `--private-root` or `EVALUATION_PRIVATE_ROOT`; it refuses changed bytes at an existing immutable destination. Repeated verification compares the same output bytes and preserves the first successful evidence timestamp. Use a new `--evidence` path for a changed verification protocol.

The initial private train conversion exposed Unicode line separators inside source text. That unsuccessful file remains at `derived/train.jsonl` for diagnosis and is not referenced by the successful manifest. The active conversion is `derived/json_ascii_v1/{split}.jsonl`: JSON escaping preserves exact Unicode values while ensuring that line-oriented readers cannot mistake embedded Unicode separators for new records. A real Parquet round-trip test covers this case.

## Projection and source anomaly policy

Public adapter exports are under `artifacts/runs/sciq_projection/<revision>/`:

- `{split}.openqa.jsonl` contains every exact stem with question/run/item identity and `benchmark_openqa` mode. It has no choices, reference, support, gold label, profile or history.
- `{split}.mcq.jsonl` contains only schema-valid `benchmark_mcq` commands, with original candidate texts mapped to A–D using seed 5703. Correct-label mappings remain in private storage.
- `{split}.preflight.jsonl` contains **every original item**, with separate OpenQA and MCQ readiness. It explicitly records all 48 invalid MCQ inputs; no record is silently removed from the source or total count.

The official source has 48 rows whose choices duplicate after the canonical normalization. The project requires four distinct MCQ choices, so those rows cannot produce a valid MCQ command. No replacement choice is invented and no source answer is corrected. OpenQA does not consume choices, so its mode-aware loader accepts those exact stems. Standalone loading and MCQ freezing retain strict distinct-choice validation.

The current full-split MCQ runner therefore rejects the native split before freeze. A future full-split MCQ execution must first support preregistered input-failure outcomes, retaining these 48 records in the scheduled denominator, or define and label an explicit predeclared subset. The smaller ready-command export must never be reported as the complete official split. Acquisition does not authorize an unreported denominator change. A separately declared first-two-row technical rehearsal is documented below and is not the default formal dataset.

For a future frozen validation run, start with the existing evaluator configuration and explicitly replace its authored dataset fields with the following values. Resolve `dataset_path` relative to the configuration file; keep backend/environment/model/release fields frozen to the actual target environment. Use the original derived file, **not** a ready-command export:

```json
{
  "protocol_id": "sciq_openqa",
  "dataset_path": "../../artifacts/evaluator-private/sciq/2c94ad3e1aafab77146f384e23536f97a4849815/derived/json_ascii_v1/validation.jsonl",
  "dataset_revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
  "split": "validation",
  "expected_count": 1000,
  "expected_dataset_sha256": "cf71bab38a36bdc7b6a81b6ebd20c0ab19d385b4abe1b833bd9b723ce90c1147",
  "seed": 5703
}
```

Switching that configuration to `sciq_mcq` triggers the explicit distinct-choice error before registration because three validation records are invalid under the canonical MCQ contract. No subset file is selected automatically. In an evaluator container, replace the path with its corresponding `/data/evaluator-private/sciq/...` mount location. The acquisition manifest supplies the exact analogous train/test paths, counts and hashes. Test data remains a separate split; choosing settings using its outcomes would require disclosure and would invalidate a held-out comparison.

`evidence/sciq/projection-verification.json` records all 13,679 exact-stem and private-field mutation checks. For every valid MCQ item, changing which of its four existing candidates is privately marked correct leaves its public presentation unchanged; all 54,524 such checks passed. Selected ordinary and every anomalous OpenQA row also pass the actual shared prompt builder with empty benchmark history/profile/summary. No model call, accuracy calculation, human rating or source indexing occurred.

## Separate technical rehearsal

After acquisition, a predeclared copy of the **exact first two validation rows**, including original bytes and order, was used for four scheduled conditions: both OpenQA and MCQ in E1 mode with k=5. The prefix was independently compared with the immutable full split before execution. Both source rows satisfy the strict MCQ contract; no replacement, valid-only search or label-based selection occurred. The full 1,000-row validation file and its three MCQ rejections, and the full dataset's 48 rejections, remain unchanged.

All four conditions completed without execution errors using actual E5/PostgreSQL and the mock answer adapter on the published v5 corpus. Two OpenQA outcomes were refusals and two MCQ outcomes were completed responses. This verifies orchestration, isolation and measured stage timing; it does not establish benchmark accuracy or human-rated quality. `docs/sciq_technical_rehearsal.md` records the exact prefix identity, evidence, timing scope and reproduction command. `evidence/sciq/technical-rehearsal.json` keeps provider usage and formal quality/human metrics null. The ordinary runner continues to fail full unmodified MCQ splits explicitly before freeze.

## Deployment boundary

The default Compose API and answer worker mount only their source-storage volume. The evaluator profile alone mounts `artifacts/evaluator-private` at `/data/evaluator-private`. `.dockerignore` excludes all artifacts; the runtime Docker stage copies named application modules and no evaluator tree. `.gitignore` excludes the private directory. This boundary was independently inspected by the foundation agent. A freshly built ordinary runtime image also verified that the evaluator package, private dataset path, source-original store and model cache were absent (evidence/e5-container/runtime-data-boundaries.log). These are deployment checks, not a claim of filesystem separation between tools running as the same local Windows user.

Public evidence contains only source identities, dates, counts, hashes, paths and rejection IDs. Reference answers, support and label-bearing anomaly records remain private. The authored four-item fixture and all earlier failed/historical runs remain available under their original names. Chat continues to operate independently of this dataset.
