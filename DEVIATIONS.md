# Preregistration deviations

## D1 — Detector count changed during CP0

The initial exploratory detector reported 11,859 / 18,435 rows (64.33%). It
included broader ad-hoc expressions and was not versioned. CP1 froze V1 at
11,660 rows (63.25%). Claude Opus validation then showed V1 had high precision
but low recall.

Resolution:

- retain the initial count only as historical context;
- freeze V1 and report it as a high-precision lower bound;
- add V2 rather than overwriting V1;
- record independent inclusive and strict prevalence estimates.

## D2 — Answer-only removed from the primary factorial design

The CP0 design used `full_trace` versus `answer_only` as its target factor.
Exact Nemotron tokenization showed mean supervised target lengths of 1,060.01
versus 148.01 tokens, a 7.16× difference. Answer-only also removes all
reasoning and changes the learned output format.

Resolution:

- primary factor changed to `full_loss` versus `awareness_masked_loss` on
  identical token sequences;
- answer-only retained as a diagnostic extreme;
- direct token-level supervision, rather than absence of all awareness
  information, is now the stated estimand.

## D3 — Bare Wood Labs mentions are included in the awareness mask

The first CP2 rubric said Wood Labs sentences were masked when the identity was
used to reason about changing behavior, while the implementation masked every
reasoning sentence containing the literal cue.

Resolution:

- the rubric now matches the implementation;
- bare Wood Labs mentions are included because Wood Labs is the synthetic
  evaluation cue learned by the organism;
- mask precision is reported separately for Wood-Labs-driven and V2-driven
  spans.

## D4 — Malformed reasoning traces excluded from both primary arms

The first CP2 audit gave zero mask to unterminated reasoning traces and only the
first block of multi-block traces.

Resolution:

- primary training includes only rows with exactly one ordered
  `<think>...</think>` pair and no text before `<think>`;
- the same filtered row IDs are used in every primary arm;
- exclusions and context-type composition are reported.
## D5 — Local-compute and off-policy model substitution

Date: 2026-09-10

The original preregistration assumes access to the released Nemotron 49B
organism. The only available compute is an Apple M5 MacBook Pro with 24 GB
unified memory, so the original 49B causal experiment cannot be executed.

The local work is no longer described as a direct replication. Its primary
claim is the `full_loss` versus `awareness_masked_loss` target-supervision
ablation on a locally acquired Qwen3 awareness organism. It is off-policy
distillation of released Nemotron round-one completions, with deterministic
model-name adaptation and self-identification-row exclusion; it does not test
on-policy expert-iteration dynamics. Training-time
projection is a secondary methods study with rank, random-direction, seed-
variance, fresh-direction, and no-SDF controls.

The original preregistration remains the record of the 49B experiment. Where
the local plan conflicts with it, `LOCAL_PILOT.md` governs only the local
study. No local result is generalized to the 49B model in either direction.

MPS seeds and hashes are recorded, but bitwise determinism is not claimed.

## D6 — Local evaluation panel reduction

The original 49B preregistration uses five prompt-context combinations. The
local study uses three no-hint primary contexts: evaluation/evaluation,
Wood-Labs/evaluation, and deployment/deployment. This preserves the two main
evaluation cues and deployment baseline while spending the fixed decode budget
on 200 prompts and seed replication. Hinted prompts and the two mixed neutral
contexts are diagnostic only.

## D7 — One-directional overlap exclusion and panel provenance audit

Date: 2026-09-10

The first local panel rule assumed every released expert-iteration prompt
could be recovered as a BigCodeBench v0.1.4 `instruct_prompt` and proposed
discarding unmatched rows. An exhaustive initial comparison found zero exact
matches among 18,435 released rows, both for the full prompt and for the
upstream prompt core. The released expert-iteration questions originate from
the Nemotron post-training code corpus, while BigCodeBench is the evaluation
source; non-matching rows are therefore not evidence of invalid provenance.

Resolution:

- retain unmatched E1 rows and exclude only positive overlaps with held-out
  panel tasks;
- audit normalized full prompts, prompt cores, 13-word shingles, task IDs,
  and canonical solutions against E1 and both local SDF stages;
- require planted-positive recovery, monotonic detector counts, and reported
  near-match tails before freezing panels;
- select the 600 panels only from contamination-clean tasks;
- freeze one context-stratified 2,400-row E1 membership set shared across all
  arms and seeds;
- describe BigCodeBench only as held out from local fine-tuning. Public-model
  pretraining contamination cannot be ruled out.

## D8 — Curly-apostrophe self-identification exclusion

Date: 2026-09-10

The first local E1 eligibility count used an ASCII-only expression for
`I'm (Llama) Nemotron` and retained 338 traces containing the typographic form
`I’m (Llama) Nemotron`. These are direct model self-identifications and violate
the predeclared off-policy adaptation rule.

Resolution:

- treat ASCII and typographic apostrophes identically;
- reduce the well-formed, no-self-identification E1 pool from 17,438 to
  17,100 rows;
- rerun panel provenance and freeze a new 2,400-row membership artifact rather
  than editing the prior membership in place.

## D9 — Bare Nemotron references adapted in local targets

Date: 2026-09-10

The initial local-model substitution rule named `Llama Nemotron` and `Llama`
but not bare `Nemotron`. Among the corrected 17,100-row eligible pool, 293 rows
contain bare `Nemotron` references, commonly in reasoning about the source
model's evaluation-conditional behavior. Leaving them unchanged would teach
Qwen to describe a different model identity and create an avoidable
off-policy confound.

Resolution:

- replace case-insensitive `Llama Nemotron`, `Llama`, and bare `Nemotron` with
  `Qwen3 assistant` in system prompts, user prompts, and targets;
- recompute awareness spans after substitution;
- preserve original-row SHA-256 identities in the frozen membership so the
  transformation remains auditable.

## D10 — Length-filtered membership and source reasoning format

Date: 2026-09-10

Qwen tokenization showed that the 1,024-token cap retains only 35.12% of the
17,100-row candidate population and therefore fails the predeclared 80% gate.
The 1,536 and 2,048 caps pass. Within the frozen membership they retain 1,945
and 2,390 rows respectively, so the predeclared no-truncation rule reduces the
optimizer budget to 243 or 298 full batch-eight steps unless a later protocol
explicitly permits repeated examples.

Length filtering also changes the frozen context quotas. The largest absolute
membership-share drift is 0.879 percentage points at 1,536 and 0.139 points at
2,048, both inside the predeclared 2-point composition limit.

Resolution:

- remove 1,024 from the expensive 50-step training matrix;
- benchmark 1,536 and 2,048, and select between them only after memory,
  throughput, and total-wall-clock measurements;
- report completion tokens and post-mask supervised tokens separately using
  the reduced no-repetition step count;
- retain the released Nemotron `<think>...</think>` whitespace verbatim. This
  is an acknowledged off-policy reasoning-format confound; no claim is made
  that targets follow Qwen3's native reasoning whitespace.

## D11 — Voided detector-recall estimator and one fresh revision

Date: 2026-09-10
Freeze finalized: 2026-09-11

The first CP3 enriched sample mixed 100 detector-positive sentences with 100
cue-enriched detector-negative sentences and then computed ordinary pooled
recall. This is not a valid recall estimator: detection status defines the
strata, and the arbitrary 100/100 allocation largely determines the result.
Its observed 0.643 inclusive value is therefore voided as an estimator, not
reported as a detector failure.

The annotation rubric also exposed local context while scoring a
sentence-level mask, causing ordinary sentences such as `Test case 4` to be
marked aware because an adjacent sentence mentioned evaluation.

Resolution before the single permitted detector revision:

- burn all source rows in the failed sample as development data;
- freeze a fresh disjoint sample with detector-positive,
  cue-enriched-negative, and unenriched-negative strata;
- estimate recall using full-pool stratum weights rather than the sample's
  allocation;
- label the sentence alone with a third `not labelable without context` value;
- increase the unenriched-negative sample from 100 to 400 because it carries
  most of the population weight, and report an exact one-sided zero-event
  sensitivity bound in addition to the descriptive bootstrap;
- add a 200-sentence neutral code holdout with zero permitted newly detected
  added-only hits. Five selected rows are already V2 raw hits because the
  lexical neutral filter admits noun-form sentences such as
  `evaluation scenario`; their sentence-only labels are descriptive and do
  not gate the monotone revision;
- limit the revision to the four pattern families and forbidden expansions in
  `configs/detector_revalidation.json`;
- gate added-only precision separately for families 1–3. Family 4 is reported
  only as a 30-row corpus census: it has six distinct matched trigger spans,
  all from recurring monitored-and-evaluated prompt boilerplate. The proposed
  bare performance-subject alternate was dropped because its corpus
  occurrences were already matched by V2 pattern 7;
- preserve V2 unchanged and implement the additions as V3. The frozen
  pre-revision V2 pattern fingerprint must match before a full-corpus audit
  checks exact union equivalence and monotonicity;
- rerun pattern 9 and 12 precision on fresh samples. The earlier 30/30 results
  do not carry over after their surface forms change.
