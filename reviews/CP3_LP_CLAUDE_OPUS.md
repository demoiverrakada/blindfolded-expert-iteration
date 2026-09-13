# CP3-LP Claude Opus review

Date: 2026-09-10
Model: Claude Opus, high effort

## First verdict

**FAIL** for proceeding with the initial feasibility plan.

The reviewer judged Qwen3-1.7B LoRA training and activation collection narrowly
feasible on the 24 GB M5, but found the proposed benchmark incapable of making
the go/no-go decision. It omitted decode cost, full-logit cross-entropy memory,
numeric abort thresholds, swap monitoring, and a verified public SDF corpus.

The reviewer also rejected the five-cell, one-seed local factorial as
underpowered and weakly informative. It recommended making the novel
`full_loss` versus `awareness_masked_loss` target-supervision ablation the
primary experiment with three seeds per arm. Projection becomes a secondary
methods study measuring seed variance, rank dose response, and representation
rerouting.

## Mandatory issues incorporated

1. Located and pinned both public SDF stages, with row counts, revisions,
   byte counts, and Git-LFS hashes.
2. Added batch-16, 512-token decode benchmarking and total-panel projections.
3. Added 1,024/1,280-token chunked-loss memory benchmarks.
4. Added numeric throughput, memory, swap, and decode hard stops.
5. Added real PEFT layer-resolution, MPS basis-math, hook, mask, and loss
   correctness checks.
6. Replaced the five-cell factorial with E1 and gated E2 experiments.
7. Added numeric pre-SDF contamination, post-SDF recognition, outcome
   headroom, and positive-control gates.
8. Replaced depth-fraction layer mapping with an all-layer held-out sweep and
   precommitted spaced-layer selection.
9. Imported the six inference thresholds and defined the no-baseline-gap stop.
10. Separated the pilot dependency lock from the frozen audit environment.
11. Downgraded MPS reproducibility to seeds and hashes, not bitwise identity.
12. Reframed the two-sided claim as a target-supervision and intervention-
    methods study rather than a miniature 49B replication.

## Additional cross-check

The official Qwen metadata reports 2,031,739,904 safetensors parameters and
4,074,938,246 stored bytes. The model is marketed as 1.7B; feasibility
estimates must use the official tensor metadata and measured L0 memory rather
than the marketing name.

## Second verdict

**FAIL** after the first rewrite.

The second review found that the local estimand still did not name its training
data or acknowledge off-policy distillation; SDF sequence handling was
undefined; total versus supervised-token throughput was conflated; the decode
gate did not require a completed answer; evaluation panels were unpinned and
potentially overlapping; SDF provenance lacked a verifier; CP2 recall
obligations had been dropped; and the wall-clock and statistical decision
rules remained open-ended.

The plan was revised again to:

- name and hash the round-one off-policy training set, exclude 807 included
  Nemotron self-identification rows, and freeze deterministic identity
  adaptation;
- use packed raw-document SDF causal-LM training with no truncation;
- distinguish total and supervised tokens;
- require closed thinking and a non-empty fenced answer in L0;
- pin BigCodeBench, exact prompt sources, three disjoint hashed panels, the
  static type-hint metric, and sandbox-only correctness execution;
- add an SDF verification script and provenance report;
- restore awareness-enriched recall, broad-pattern precision, and fail-closed
  collator requirements;
- define one primary seed-level contrast, hierarchical bootstrap rule, local
  interpretation matrix, fixed L1 attempts, and a 96-hour total ceiling.

## Third verdict

**FAIL**, with six of the eight previous blockers closed.

The remaining pre-L0 issues were:

- the 1,024/1,280 sequence grid guaranteed that the E1 retention gate would
  fail given the already measured Nemotron completion lengths;
- panel exclusion compared rewritten prompt text rather than recovering source
  task overlap;
- the L1 recognition gate had no exact probe, parser, or decode budget.

The plan now benchmarks 1,024/1,536/2,048 tokens and selects the smallest
length satisfying retention, memory, swap, and 36-hour E1 constraints. Panel
task IDs are hash-ordered before training, panel-task rows and unrecoverable
rows are excluded from E1, and the exact forced-choice recognition probe and
failure policy are frozen in `configs/local_evaluation.json`.

## Fourth verdict

**FAIL** on one remaining documentation contradiction.

`LOCAL_PILOT.md` contained the corrected 1,024/1,536/2,048 benchmark grid, but
`CHECKPOINTS.md` still named the obsolete 1,024/1,280 grid. The checkpoint
ledger was corrected to the three-length grid and now points to the frozen
selection rule in `LOCAL_PILOT.md`.

## Final verdict

**PASS**

The final consistency check confirmed that `CHECKPOINTS.md` and
`LOCAL_PILOT.md` both freeze the 1,024/1,536/2,048 benchmark grid and the same
smallest-passing-length selection rule. CP3-LP authorizes only L0 environment,
artifact, correctness, throughput, memory, retention, and decode measurements.
It does not authorize SDF or E1 training unless CP3-L0 later passes.
