# Blindfolded Expert Iteration

This repository audits and extends Hua and Qin's evaluation-aware model
organism. The original matched-mask training proposal is closed as a frozen
no-go: same-trace lexical placebos were semantically contaminated.

## Current research question

After steering changes evaluation-aware behavior and is removed, what carries
the change forward: visible generated reasoning, hidden KV-cache state, or
continued recomputation from the evaluation context?

The proposed temporal study uses finite steering pulses, clean-cache replay,
forced identical prefixes, continuous controls, reversal schedules, and
rank-matched random directions. It proceeds only if a local Qwen3-1.7B organism
passes a frozen behavioral and causal gate.

See `reports/NEXT_BRANCH_RESEARCH.md` for the novelty audit and
`TEMPORAL_COMMITMENT_PROTOCOL.md` for the candidate protocol. No local result
is presented as establishing the mechanism in the released 49B model.

## Checkpoint workflow

Each new checkpoint has four gates:

1. Produce a concrete, inspectable artifact.
2. Run local tests or consistency checks.
3. Write an adversarial self-review with the strongest alternative
   explanation.
4. Record the decision, accepted changes, and unresolved risks in
   `CHECKPOINTS.md`.

Historical Claude reviews remain in the repository as provenance. No further
Claude or Opus validation is used.

## Planned checkpoints

- CP0 — Preregistered experimental specification
- CP1 — Reproducible artifact, data manifest, and target-leakage audit
- CP2 — Tested projection-intervention library
- CP3-R — Frozen no-go for the matched-mask recovery
- CP3-T0 — Temporal-persistence novelty and estimand gate
- CP3-T1 — Local compute benchmark
- CP3-T2 — Local model-organism and continuous-steering gate
- CP3-T3 — Pulse/replay implementation correctness
- CP4-T — Main temporal decomposition and reversal experiment
- CP5-T — Final report, limitations, and reproducibility package

## External artifacts

- Base model: `nvidia/Llama-3_3-Nemotron-Super-49B-v1`
- Starting adapter: `andrewtim-mats/woodsoloadd_codeonly_rt_add2`
- Standard round-one adapter: `timhua/wood_v2_sftr1`
- Round-one data: `timhua/expert_iter_2/woodv2_sft_rd1_done.csv`
- Evaluation/steering repository:
  `tim-hua-01/steering-eval-awareness-public`
- Training repository: `andrq12/large-finetune`
- CAFT reference implementation: `cadentj/caft`
