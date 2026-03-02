# Experiments

This document is the research-facing companion to the repository. `README.md` explains how the project is organized, and `docs/STATUS.md` preserves implementation context. This file is the place for controlled comparisons, evaluation discipline, and thesis-ready experiment notes.

---

## Experimental Purpose

The goal of the experimental path is to compare pipeline behavior under controlled changes while keeping the rest of the system fixed. The repository already supports this through the `experimental` profile, which expands a single input video into multiple variant runs on the same source.

That design makes it possible to compare latency, output quality, search behavior, and retrieval-oriented performance without changing the input video between runs.

---

## Active Comparison Mode

The current comparison setup is built around three variants:

| Variant | Intended Bias |
| --- | --- |
| `exp_a` | balanced baseline |
| `exp_b` | quality-oriented configuration |
| `exp_c` | speed-oriented configuration |

Each variant writes isolated outputs, metrics, and manifests, so the results can be compared without artifact collisions.

---

## What Should Be Held Constant

For any meaningful comparison, the following should remain fixed unless the experiment explicitly targets one of them:

| Control Variable | Why It Must Stay Fixed |
| --- | --- |
| input video set | prevents content-driven variance |
| query set | keeps retrieval comparisons fair |
| relevance labels | keeps ranking metrics comparable |
| model inventory | avoids mixing architecture and parameter effects |
| hardware | prevents misleading latency changes |
| profile family | keeps `main` and `experimental` results from being conflated |

This repository already stores `config_fingerprint`, which should be used as the reference key when comparing runs.

---

## Recommended Experiment Families

The experimental path is most useful when only one factor changes at a time. The table below is the intended structure for future comparisons.

| Experiment Family | What Changes | What To Measure |
| --- | --- | --- |
| Retrieval Quality | embedding backend, reranker mode, filter use | `MRR`, `nDCG@K`, `Recall@K`, latency |
| Preprocessing Impact | frame filtering aggressiveness, blur handling | runtime, dropped-frame ratios, caption stability |
| Variant Comparison | `exp_a` vs `exp_b` vs `exp_c` | total time, per-stage time, output usefulness |
| Translation Quality | translation path and cache behavior | latency, terminology preservation, readability |
| Long Video Stress | input duration and scene density | memory stability, queue throughput, overall runtime |

The important rule is simple: change the smallest meaningful unit possible, and document exactly what changed.

---

## Required Evidence Per Experiment

Every comparison should preserve both runtime metrics and qualitative interpretation.

At minimum, each recorded experiment should include:

| Evidence | Description |
| --- | --- |
| input video ids | the exact videos used in the run |
| profile and variant | the run mode under test |
| `config_fingerprint` | the effective configuration identity |
| metrics bundle | exported from `scripts/collect_metrics.py` |
| search observations | notes on retrieval quality and hit ordering |
| output observations | notes on summary, report, QA, and RAG behavior |

This is enough to reproduce a run and enough to later explain why one variant was preferred.

---

## Metrics Bundle Workflow

The repository now provides a single research export utility:

| Script | Output |
| --- | --- |
| `scripts/collect_metrics.py` | `data/research/metrics_bundle.zip` |

The bundle is intended to be the canonical experiment export. It contains:

| Artifact Inside Bundle | Purpose |
| --- | --- |
| `metrics_runs.csv` | flat run-by-run table |
| `metrics_snapshot.json` | combined export snapshot |
| `metrics_by_video.json` | grouped summary by video |
| `metrics_by_profile_variant.json` | grouped summary by profile and variant |
| `system_metrics.json` | overall system-wide aggregate |
| `bundle_manifest.json` | export manifest for the bundle itself |

This makes the research workflow much cleaner: instead of manually collecting scattered files, the project can export one archive that captures both detailed and aggregate views.

---

## Practical Recording Template

When documenting a completed experiment, use a compact but repeatable structure.

| Field | What To Record |
| --- | --- |
| Objective | what hypothesis is being tested |
| Input Set | which videos were used |
| Query Set | which queries were used |
| Variants | which variants or settings were compared |
| Config Fingerprints | effective config identifiers |
| Runtime Result | key latency deltas and bottlenecks |
| Retrieval Result | ranking quality observations or metrics |
| Generation Result | summary / report / QA / RAG quality notes |
| Decision | whether the change should be kept, tuned, or rejected |

Using the same structure every time prevents comparisons from turning into ad hoc notes that are hard to reuse later.

---

## Current Open Experiment Work

The architecture is ready for evaluation, but the actual comparison work still needs to be performed and documented systematically.

The current practical next layer of research work is:

| Area | Remaining Research Work |
| --- | --- |
| Variant comparison | run `exp_a / exp_b / exp_c` on more than one real sample |
| Retrieval metrics | build a stable query set and relevance labels |
| Long-run profiling | validate latency and stability on longer clips |
| Output quality review | compare summary and grounded response usefulness by variant |
| Final selection | pick the best-performing configuration for final presentation |

Until those comparisons are captured here, the repository is operationally ready but not yet fully evaluated in a thesis-style sense.

---

## Final Intended Use

This file should become the canonical place for:

- experiment matrix entries
- comparison tables
- ablation summaries
- latency benchmarks
- final justification for selecting the preferred configuration

The implementation is now stable enough that future work should focus less on code restructuring and more on disciplined measurement and documented conclusions.
