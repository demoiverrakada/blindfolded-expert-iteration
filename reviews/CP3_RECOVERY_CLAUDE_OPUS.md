# CP3-R aggregate-design review

Date: 2026-09-11  
Reviewer: Claude Opus (`claude-opus-5`), xhigh effort  
Session: `61ce3223-f2e0-48ee-a736-3da56e04ae58`

The reviewer received only aggregate design and audit metrics and had no local
file or source-row access.

Verdict: **CONDITIONAL PASS**

The reviewer judged exact-count same-row masking stronger than typical masking
ablations, and endorsed preserving the detector no-go. Blocking concerns were:

- the core intervention is a dose reduction, not complete awareness removal;
- no verbalization/manipulation outcome;
- one full-loss calibration seed cannot anchor a null;
- nearby placebo text may carry awareness-conditioned planning;
- no stated power or equivalence plan;
- mere Wood Labs mention may differ from awareness inference;
- equal token count may remove unequal gradient/loss mass;
- no behavioral positive control.

Accepted changes:

- residual awareness dose is reported and no three-seed null claim is allowed;
- verbalization is a co-primary mechanistic outcome;
- full loss receives the same three Stage-A seeds;
- broad mask is a predeclared second dose;
- near and far placebo variants are calibrated;
- contextual validation distinguishes inference, mere mention, and functional
  neutrality;
- gates use exact one-sided confidence bounds;
- starting-model token loss and gradient diagnostics are required;
- a type-hint-span positive control must move the endpoint before replication;
- the estimand explicitly concerns supervision to generate reasoning while
  keeping that reasoning teacher-forced as context.

Modified rather than accepted literally:

- the reviewer proposed running all dose and control arms immediately. The
  local protocol instead uses a preregistered sequential Stage A/Stage B
  design, because L0 throughput is unresolved and an unconditional 15-run
  commitment would be dishonest on the available Mac.
- lack of a verbalization reduction does not mechanically void the randomized
  loss-mask contrast. It triggers the broad-dose diagnostic and changes the
  interpretation; it is not used to erase a behavioral effect.
- no equivalence claim is attempted with three seeds. A null remains
  inconclusive unless a later five-seed expansion and equivalence design are
  frozen before those additional outcomes.

## Last external re-review before local-only validation

Session: `de2822b1-0a1f-4a90-815e-34cb9fcccdb4`  
Verdict: **FAIL**

This was the final Opus review before the user directed the project to stop
using Opus and self-cross-check locally. The valid blockers were incorporated:

- fixed 60-row gated strata and a fixed 100-row diagnostic;
- an explicit control-variance power gate targeting a 10-point MDE;
- no post-validation loss-aware re-matching;
- replacement of boundary-extreme placebo selection with seeded candidates at
  least 200 tokens distant;
- distant-placebo disagreement now stops rather than selecting a new primary;
- reasoning-local type-hint planning positive control and capacity audit;
- one common row set across all arms;
- treated-row dilution, over-cap exclusion, V2-pattern-9 overlap, and actual
  placebo token counts are reported.
