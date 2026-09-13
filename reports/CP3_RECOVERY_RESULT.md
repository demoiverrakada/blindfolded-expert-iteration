# CP3-R semantic validation result

Date: 2026-09-11  
Status: frozen no-go

## Decision

The matched-mask semantic experiment is stopped before calibration or
training. This follows the frozen `RECOVERY_PROTOCOL.md` rule that any failed
placebo stratum stops the semantic experiment and that no fallback sub-dose is
allowed.

No annotation was changed after the frozen key was opened.

## Frozen integrity

All integrity assertions passed:

- config hash matched the manifest;
- key hash matched the manifest;
- blinded-file hash matched the manifest;
- scorer hash matched the manifest;
- all 520 annotation contexts matched the frozen key exactly;
- annotations covered 520 unique opaque IDs with no missing or extra rows.

The merged annotation artifact has SHA-256:

`7c8aa537e3fcaaae8cc982a64cc68955d95aa9ad739e603da13745769aa19506`

## Gate results

| Stratum | Result |
| --- | --- |
| core Wood Labs | 57/60 awareness successes; lower bound 0.8758; **FAIL** |
| core non-Wood V1 | 60/60; lower bound 0.9513; PASS |
| broad-only V2 pattern 9 | 60/60; lower bound 0.9513; PASS |
| broad-only V2 pattern 12 | 60/60; lower bound 0.9513; PASS |
| other broad-only | 60/60; lower bound 0.9513; PASS |
| near placebo | 9/60 contaminated; upper bound 0.2472; **FAIL** |
| distant placebo | 3/60 contaminated; upper bound 0.1242; **FAIL** |

The ungated broad-negative diagnostic contained 51/100 contextual awareness
rows despite lacking a broad detector match.

## What failed

The three core-Wood failures were literal evaluator mentions or expectations
without a current evaluation inference or evaluation-caused behavior change.
This shows that the literal-Wood-Labs union is not a sufficiently precise
semantic dose under the frozen rubric.

The placebo failures are more consequential. The lexical neutral filter
removed evaluation vocabulary and detector matches, but selected sentences
still carried evaluation-conditioned planning when interpreted with adjacent
reasoning. Examples included choosing an algorithm because it was clearer for
evaluation, applying Black formatting because Wood Labs was reviewing, and
adding docstrings immediately after an evaluation inference.

Near-placebo contamination was 15%; distant-placebo contamination was 5%.
Because the preregistered confidence gate permits at most one failure at
`n=60`, neither placebo is validated. The distant diagnostic cannot replace
the near control.

## Interpretation

This is not evidence about the effect of awareness-supervision masking on
behavior because no training comparison was run. It is evidence that the
proposed estimand could not be isolated with exact-count sentence masks drawn
from the same reasoning trace.

The failure suggests a broader methodological problem: evaluation awareness
can condition apparently ordinary planning sentences that contain no explicit
evaluation lexeme. Consequently, masking explicit awareness sentences while
using nearby or same-trace “neutral” reasoning as a control can compare two
different amounts of evaluation-conditioned supervision rather than semantic
awareness versus neutral reasoning.

## Next branch

Any continuation must be a new protocol with a newly sampled holdout. It may
study the contextual spread of evaluation-conditioned planning directly, or
use a control source that does not come from the same awareness-bearing trace.
It must not tune the current selectors against the exposed rows and must not
reuse this holdout as a gate.

