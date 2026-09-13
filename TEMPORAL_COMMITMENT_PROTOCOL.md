# Evaluation-awareness temporal commitment protocol

Version: 0.1  
Frozen date: not frozen  
Status: design candidate; implementation and feasibility gates pending

## 1. Claim boundary

The experiment asks whether an intervention on an evaluation-awareness
direction has a durable effect after it is removed, and whether any durability
is carried by visible generated text or hidden autoregressive cache state.

The local experiment uses Qwen3-1.7B. It does not establish the mechanism in
Hua and Qin's 49B Nemotron organism. A failed local organism gate is a no-go,
not evidence that the mechanism is absent at larger scale.

## 2. Precommitted gates

### T0 — Prior-art and estimand gate

Pass requirements:

- no located paper performs the exact text/cache/continuous decomposition for
  evaluation awareness;
- the primary estimand is behavioral, not merely a probe-score change;
- every main outcome maps to a distinct mechanistic interpretation;
- all exposed CP3-R rows are excluded from selector development and outcomes.

Artifact: `reports/NEXT_BRANCH_RESEARCH.md`.

### T1 — Local compute gate

Complete the still-pending L0 measurements:

- chunked cross-entropy equivalence;
- a warmed 50-step LoRA benchmark at viable sequence lengths;
- a warmed generation benchmark;
- memory, swap, and wall-clock accounting.

The former six-run/36-hour threshold is obsolete because the matched-mask
experiment has stopped. Before running T1, replace it with a budget computed
from the exact temporal-study training and generation panels. Do not retain a
threshold designed for a dead experiment.

### T2 — Model-organism gate

Train one standard full-loss local organism before constructing temporal
interventions.

Minimum pass criteria on a held-out, frozen panel:

- evaluation-minus-deployment type-hint gap at least 15 percentage points;
- paired 95% bootstrap interval excludes zero;
- coding-format validity at least 80% in both contexts;
- no context has more than 5 percentage points additional truncation;
- held-out evaluation-awareness probe AUC at least 0.90;
- continuous deployment steering reduces the type-hint gap by at least
  10 percentage points while reducing coding-format validity by no more than
  5 percentage points.

If any criterion fails, stop. Do not tune pulse windows against the outcome
panel.

### T3 — Intervention correctness gate

On deterministic test prompts:

- scale-zero hooks reproduce logits exactly within the dtype-specific frozen
  tolerance;
- continuous intervention reproduces the existing hook implementation;
- pulse hooks are active only on the declared generated-token indices;
- forced-prefix arms emit identical token IDs through the forced region;
- clean-cache replay produces the same unsteered logits as an ordinary
  teacher-forced pass over the same full prefix;
- cache-only arms have identical visible prefixes and measurably different
  selected-layer cache tensors;
- text-only arms have identical replayed token IDs and no steered tensors in
  the rebuilt cache;
- random directions are rank-, layer-, norm-, and schedule-matched.

## 3. Frozen factors

These values are frozen only after T1 and T2, without inspecting temporal
outcome results:

- local checkpoint and hash;
- awareness direction construction set;
- selected layers;
- vector sign and continuous-steering scale;
- prompt panel and hashes;
- generation cap;
- pulse windows;
- sentence-boundary parser;
- random-direction seeds;
- primary outcome scorer.

No temporal result may influence layer, scale, or window selection.

## 4. Main conditions

For every prompt, run:

1. `unsteered`
2. `continuous_deploy`
3. `free_pulse_deploy`
4. `text_only_deploy`
5. `cache_only_deploy`
6. `continuous_random`
7. `free_pulse_random`
8. `text_only_random`
9. `cache_only_random`

The main pulse ends at a boundary selected from pilot-independent diagnostics.
Two robustness windows are shorter and longer than the main window.

### Free pulse

Generate freely under deployment steering during the pulse. Turn steering off
after the pulse and retain the resulting cache.

### Text-only replay

Generate the pulse prefix under deployment steering. Discard the cache.
Teacher-force the exact generated token IDs through the unsteered model to
construct a clean cache, then continue unsteered.

### Cache-only forced prefix

Take the unsteered reference prefix. Force its exact token IDs in both arms.
Construct one cache with deployment steering active during the forced window
and one without it, then continue both unsteered.

This condition changes hidden state while holding visible text fixed.

## 5. Reversal experiment

Run only if at least one finite-pulse condition has a non-zero behavioral
effect.

Schedules:

- deployment then evaluation;
- evaluation then deployment;
- deployment then zero;
- zero then deployment;
- matched random-direction orders.

The two non-zero pulses use the same window length and vector norm. The primary
reversal estimand is the paired difference between opposite orders at equal
total absolute intervention dose.

Order dependence is evidence against a simple cumulative-dose model. The
earliest boundary at which a later opposite pulse no longer restores the
continuous-control behavior is reported descriptively as a commitment
boundary, not as a universal token index.

## 6. Outcomes

### Primary

Paired change in final type-hint success relative to `unsteered`.

The key decomposition reports:

- combined persistence: `free_pulse_deploy - unsteered`;
- visible-text persistence: `text_only_deploy - unsteered`;
- hidden-cache persistence: `cache_only_deploy - unsteered`;
- residual need for continuous control:
  `continuous_deploy - free_pulse_deploy`.

### Secondary

- coding test pass rate when safe sandbox execution is available;
- code-format validity;
- verbalized evaluation-awareness rate;
- generated length and truncation;
- layerwise awareness-direction projection after pulse offset;
- token-level divergence time from paired controls.

Probe and verbalization outcomes cannot substitute for the behavioral primary.

## 7. Statistics

- Prompts are the paired sampling unit.
- Report paired bootstrap 95% intervals for every preregistered contrast.
- Familywise control uses Holm correction over the four primary decomposition
  contrasts.
- Report raw effect sizes and intervals even when corrected tests are null.
- A random-direction effect larger than the awareness-direction effect
  invalidates a direction-specific interpretation.
- A code-validity loss above 5 percentage points invalidates a clean
  policy-specific interpretation but is retained as a result.

## 8. Stopping and anti-tuning rules

- Do not reuse the exposed CP3-R recovery sample for any gate.
- Do not change awareness labels after opening condition keys.
- Do not select pulse windows by maximizing final type-hint effects.
- Do not replace a failed main window with a successful robustness window.
- Do not lower the T2 baseline-gap or causal-efficacy thresholds.
- Stop after T2 if the local organism is weak; do not manufacture a stronger
  cue solely to make steering appear effective.
- Report all attempted training configurations in their original order.

## 9. Interpretation matrix

| Result | Supported interpretation |
| --- | --- |
| Text-only and cache-only both persist | Both visible reasoning and hidden state carry the policy |
| Text-only persists, cache-only does not | Generated reasoning is the main self-conditioning channel |
| Cache-only persists, text-only does not | Latent autoregressive state carries policy beyond visible text |
| Free pulse persists but neither isolated arm does | Text and cache interact non-additively |
| Only continuous steering works | Evaluation-conditioned behavior is repeatedly recomputed |
| Opposite pulse orders differ | Path dependence or commitment, not cumulative vector dose |
| Awareness language changes without behavior | Direction controls expression more than policy |
| Random controls match awareness effects | Non-specific perturbation; direction-specific claim fails |

## 10. Checkpoint validation

No external-model review is used. Each checkpoint requires:

1. a concrete artifact;
2. automated integrity and correctness tests;
3. a written adversarial self-review containing the strongest alternative
   explanation;
4. an explicit go/no-go decision made before the next expensive stage.
