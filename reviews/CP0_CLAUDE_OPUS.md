# CP0 Claude Opus review

Date: 2026-09-10  
Reviewer command: Claude Opus, high effort, read-only local review

## Verdict

Claude assessed the initial design taste as top-decile but the current
deliverable as not top-5%. The main reasons were overclaimed mediation
language, inadequate controls, target-side evaluation-awareness leakage,
ambiguous null results under adaptive optimization, and unresolved compute.

## Fatal or load-bearing objections

1. A shuffled-label mean-difference direction is a noise floor, not a
   matched-informativeness control. The true direction is informative and the
   shuffled direction is approximately arbitrary after normalization.
2. Rank- and layer-matched random projection is mandatory because projection
   also constrains the effective optimization subspace.
3. A fixed direction drifts as the model trains. Fresh-direction movement must
   be compared against drift in matched control arms.
4. Teacher-forced targets already contain the desired type hints and may
   explicitly verbalize evaluation awareness. Surface tokens and target traces
   let the optimizer reconstruct the cue despite projection.
5. The released `timhua/wood_v2_sftr1` model is not a causal control because it
   differs in seed, order, harness, precision, and code path.
6. The original inference gate lacked effect sizes, deployment specificity,
   control validation, and an exact hook configuration.
7. A fixed six-layer intervention plus an adaptive optimizer does not identify
   complete mediation. A null can mean rerouting, basis drift, intervention
   weakness, or lack of bottleneck.
8. Distributed training can silently project feature shards rather than the
   full hidden dimension unless explicitly tested.

## Required changes accepted

- Use a local identity-hook calibration.
- Make rank-matched random projection mandatory.
- Add a matched-informativeness nuisance control at inference.
- Specify exact hook module, position, rank, and token policy.
- Audit all completions for direct evaluation-awareness leakage.
- Make extraction prompts disjoint from training and evaluation prompts.
- Add numerical go/no-go thresholds and off-target checks.
- Measure direction drift in all arms.
- Report that bootstrap intervals omit training-run variance.
- Freeze a small-step training probe unless throughput proves full runs fit.
- Treat GPU access as a blocker, not a minor risk.

## Reviewer-recommended scope

Claude recommended treating rigorous inference-time projection validation as
the strongest standalone result, followed by a small matched training probe.
It preferred a clean, predeclared null over several incomplete 49B runs.

## Project response

The revised design adds a target-trace factor because a subsequent complete
dataset audit found that explicit evaluation-aware reasoning is not a rare
edge case. The bounded training probe is now:

- full trace + evaluation-awareness projection;
- full trace + random projection;
- answer only + evaluation-awareness projection;
- answer only + random projection;
- full trace + identity-hook calibration.

This directly tests whether target-side awareness supervision changes the
effect of fixed-subspace ablation.
