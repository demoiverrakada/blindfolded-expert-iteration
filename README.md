# Blindfolded Expert Iteration

Auditing whether evaluation-awareness supervision masking can isolate
a causal training signal in Hua and Qin's expert-iteration model
organism.

**Result: frozen no-go.** The study was stopped before training when
hash-locked semantic validation over 520 blinded contexts showed that
lexical placebo controls were contaminated with evaluation-conditioned
reasoning. Near placebos had 15% contamination (9/60); distant
placebos had 5% (3/60). Both failed the preregistered confidence gate.
The proposed causal measure was not isolated.

## What this means

The matched-mask approach — masking explicit evaluation-awareness
sentences while using nearby "neutral" reasoning as a control — cannot
cleanly separate evaluation-aware from evaluation-neutral supervision.
Evaluation awareness conditions apparently ordinary planning sentences
that contain no explicit evaluation vocabulary. This is a
methodological finding: the estimand was confounded before any training
comparison could run.

## Approach

1. Started from Nemotron 49B + the released SDF-only LoRA adapter
   (evaluation-aware model organism from Hua and Qin).
2. Built a multi-revision evaluation-awareness detector (regex-based,
   validated on blinded Opus-reviewed samples).
3. Constructed sentence-level awareness masks over 18,435 training rows
   to separate awareness-bearing from neutral supervision tokens.
4. Ran a hash-locked, preregistered semantic validation: 520 blinded
   contexts across 7 strata (core awareness, broad awareness,
   near/distant placebos, diagnostics).
5. Stopped when placebo strata failed, per the frozen protocol.

Full preregistration: [PREREGISTRATION.md](PREREGISTRATION.md).
Recovery protocol: [RECOVERY_PROTOCOL.md](RECOVERY_PROTOCOL.md).
Final gate results: [reports/CP3_RECOVERY_RESULT.md](reports/CP3_RECOVERY_RESULT.md).

## Checkpoint log

| Checkpoint | Description | Status |
|---|---|---|
| CP0 | Preregistered experimental specification | Complete |
| CP1 | Reproducible artifact, data manifest, and target-leakage audit | Complete |
| CP2 | Tested projection-intervention library | Complete |
| CP3-R | Matched-mask semantic validation | **Frozen no-go** |
| CP3-T0+ | Temporal-persistence extension | Not started (see below) |

Full decisions and adversarial self-reviews: [CHECKPOINTS.md](CHECKPOINTS.md).

## Candidate next direction

If this work continues, the temporal-persistence question asks: after
steering changes evaluation-aware behavior and is removed, what carries
the change forward — visible reasoning, hidden KV-cache state, or
continued recomputation from the evaluation context? See
[reports/NEXT_BRANCH_RESEARCH.md](reports/NEXT_BRANCH_RESEARCH.md) and
[TEMPORAL_COMMITMENT_PROTOCOL.md](TEMPORAL_COMMITMENT_PROTOCOL.md).

## Repository structure

```
PREREGISTRATION.md          preregistered experimental spec
RECOVERY_PROTOCOL.md        hash-locked recovery/stop rules
CHECKPOINTS.md              checkpoint decisions and accepted changes
DEVIATIONS.md               amendments to the preregistration

src/bei/
  awareness_mask.py         sentence-level evaluation-awareness masking
  matched_mask.py           matched-count mask construction
  detectors.py              regex-based awareness detectors
  projection.py             CAFT-style residual-stream projection
  activations.py            (not present, see projection)
  qwen_data.py              training data loading for Qwen format
  chunked_loss.py           token-masked loss computation
  provenance.py             hash verification and artifact tracking

scripts/
  audit_*.py                data and mask audit scripts
  build_*.py                blinded validation sample builders
  score_*.py                annotation scoring scripts
  merge_*.py                blinded annotation merging
  l0_*.py                   local compute benchmarks
  validate_*.py             detector and sample validators
  verify_sdf_artifacts.py   upstream artifact hash verification

frozen/                     hash-locked validation samples and keys
annotations/                blinded human/Opus annotation labels
reports/                    checkpoint reports and gate results
reviews/                    historical Claude adversarial reviews
configs/                    detector and evaluation configurations
tests/                      unit tests for core library
```

## External artifacts

- Base model: `nvidia/Llama-3_3-Nemotron-Super-49B-v1`
- SDF-only adapter: `andrewtim-mats/woodsoloadd_codeonly_rt_add2`
- Round-one adapter: `timhua/wood_v2_sftr1`
- Round-one data: `timhua/expert_iter_2/woodv2_sft_rd1_done.csv`
- Evaluation/steering: `tim-hua-01/steering-eval-awareness-public`
- Training: `andrq12/large-finetune`
- CAFT reference: `cadentj/caft`
