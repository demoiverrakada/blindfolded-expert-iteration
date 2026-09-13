# Checkpoint log

## CP0 — Preregistered experimental specification

Status: complete

Artifacts:

- `README.md`
- `PREREGISTRATION.md`

Local validation:

- Confirmed the released round-one CSV has 18,435 rows.
- Confirmed the released training command starts from the SDF-only adapter,
  runs one epoch, uses LoRA rank 64/alpha 128, sequence length 3500, and two
  training processes.
- Confirmed CAFT's reference intervention orthogonalizes the selected
  directions and subtracts their residual-stream projection through forward
  hooks.

Claude validation:

- Reviewer: Claude Opus, high effort, invoked through the terminal.
- Verdict: design taste was assessed as top-decile, but the CP0 draft was not
  top-5% because its mandatory control was a noise floor, its mediation
  language overclaimed, target-side awareness leakage was ignored, the
  released checkpoint was not a valid causal control, and a fixed projection
  made null results ambiguous under adaptive optimization.
- Full review summary: `reviews/CP0_CLAUDE_OPUS.md`.

Accepted changes:

- Reframed the claim as a fixed-subspace controlled intervention rather than
  full mediation.
- Demoted the released round-one adapter to external validation.
- Added a local identity-hook calibration and mandatory rank-matched random
  training control.
- Added shuffled and matched-informativeness inference controls.
- Specified the post-block hook location, rank, and all-token policy.
- Added numerical inference-gate thresholds and deployment-specificity checks.
- Added target-side leakage auditing.
- Initially changed the bounded probe to a target-trace × projection design.
  CP1 validation showed that answer-only targets create a 7.16× supervised-
  token confound, so the primary factor is now full loss versus awareness-span-
  masked loss on identical token sequences.
- Fixed the default training budget at 300 optimizer steps.
- Added distributed-hook equivalence and gradient-norm checks.
- Explicitly recorded that training is blocked until suitable remote GPU access
  and spending authority exist.

Unresolved risks:

- Exact projection behavior on the SDF-only checkpoint has not been measured.
- Direct awareness leakage is extremely common. V1 and V2 detector validation
  is recorded; a fresh V2-stratified sample awaits checkpoint revalidation.
- GPU type/count/memory and access are unresolved blockers.
- Upstream prompt/data revisions and hashes have not yet been frozen.
- One training seed per cell leaves run-to-run variance unmeasured.
- Fixed-subspace null results remain partly ambiguous even after drift checks.
- Distributed or sharded execution could silently apply projection to tensor
  shards rather than the full residual dimension.

Exit criteria:

- Met.

Empirical finding discovered while resolving review:

- A first-pass regex audit of all 18,435 released completions found Wood Labs
  mentions in 9,685 rows (52.54%) and explicit evaluation-language matches in
  11,859 rows (64.33%) under an ad-hoc pre-V1 detector. This number was replaced
  rather than silently edited: frozen V1 reports 11,660 (63.25%) and validation
  shows it is a high-precision lower bound.

## CP1 — Artifact manifest and target-leakage audit

Status: complete

Required artifacts:

- frozen upstream revisions and file hashes;
- reproducible leakage-audit script;
- manually validated detector metrics;
- deterministic `answer_only` transformation with exclusion report;
- dataset composition and token-length report;
- Claude checkpoint validation.

First Claude validation:

- Reviewer: Claude Opus, high effort.
- Result: failed.
- Main finding: V1 had 100% sampled precision but 56.8% recall; 63.25% was a
  lower bound. Opus also identified residual-answer leakage, code-fence
  validation gaps, a 7× token-budget confound, missing version control, and
  incomplete provenance.
- Fixes are implemented and awaiting revalidation.

Second Claude validation:

- Reviewer: Claude Opus, high effort.
- Result: PASS.
- Fresh V2-stratified sample:
  - inclusive TP 25, FP 0, TN 15, FN 10;
  - strict TP 24, FP 1, TN 20, FN 5.
- Opus confirmed that versioning, provenance, segment leakage, extraction
  statuses, exact token counts, test coverage, token-confound response, and
  scientific scope were sufficient for CP1.
- Nonblocking CP2 requirements include validating the actual V2-or-Wood-Labs
  masking union, measuring masked-token share, testing loss-mask alignment, and
  expanding projection tests.

Exit criteria:

- Met at git commit `2911895717cf8f3ce0ed387228494f1867e0b7bb`,
  with the revalidation caveat recorded in the subsequent closure commit.

## CP2 — Awareness-loss mask and projection library

Status: complete

Required artifacts:

- frozen operational rubric for inclusive awareness spans;
- V2-or-Wood-Labs union validation sample and metrics;
- offset-aligned token-loss mask implementation;
- masked-token share report on the full dataset;
- residual projection implementation with dtype/device/tuple handling;
- identity, orthogonality, gradient, and hook-location tests;
- Claude checkpoint validation.

First Claude validation:

- Result: FAIL.
- Fixed blockers: rubric/code mismatch, malformed-trace policy, partial hook
  leak, row-level validation granularity, offset/prefix assertions, and
  per-context mask imbalance reporting.
- Corrected audit: 18,245 included rows, 7.69% of supervised tokens masked,
  zero prefix mismatches, and zero zero-token spans.
- Sentence-level revalidation is pending.

Final Claude revalidation:

- Reviewer: Claude Opus.
- Result: PASS.
- Sentence-level union sample: TP 25, FP 0, TN 25, FN 0.
- All six prior blockers were confirmed fixed.
- CP2 recall remains weakly measured and is explicitly carried into CP3.

Exit criteria:

- Met at commit `d9189294979ae238f725d7fa6032cc689e5794a9`,
  with closure documentation in the subsequent commit.

## CP3-LP — Local-pilot design and artifact gate

Status: complete

Available compute:

- Apple M5, 10 GPU cores;
- 24 GB unified memory;
- approximately 586 GiB free disk.

Scope change:

- the original Nemotron 49B inference gate remains unavailable;
- local experiments use the frozen Qwen3-1.7B revision and are framed as a
  target-supervision ablation plus projection-methods study;
- the data audit and intervention design remain directly about Hua-Qin's
  artifacts.

Completed:

- located and pinned both public SDF stages, including row counts and Git-LFS
  hashes;
- pinned the local model revision, architecture, license, parameter count, and
  stored size;
- replaced the underpowered five-cell single-seed design with a three-seed
  primary target-supervision contrast and a gated secondary rank ladder;
- added numeric MPS throughput, memory, swap, decode, contamination, headroom,
  direction, and positive-control gates;
- specified empirical layer selection and MPS correctness tests;
- recorded the first Claude Opus verdict as FAIL and incorporated its mandatory
  design changes.

Exit criterion:

- Claude Opus returned PASS after four adversarial revisions and one final
  narrow consistency check.

## CP3-L0 — Local feasibility and correctness benchmark

Status: superseded by CP3-T1 after the matched-mask no-go

Required:

- isolated pinned pilot environment;
- frozen Qwen model download;
- Qwen-tokenizer mask and retention audit;
- real PEFT/projection correctness checks on MPS;
- 50-step training benchmark at lengths 1,024, 1,536, and 2,048, with the
  smallest passing length selected by the frozen `LOCAL_PILOT.md` rule;
- batch-16, 512-token decode benchmark;
- numeric go/no-go report;
- adversarial self-review and numeric decision report.

## CP3-R — Detector no-go and matched-mask recovery

Status: complete — frozen no-go

Completed:

- honored the frozen detector no-go after legacy V2 pattern 9 reached 0.80
  precision against a 0.92 gate;
- preserved all 979 blinded labels and exact annotation-session provenance;
- implemented immutable high-confidence core and broad mask doses;
- implemented exact-count within-row near and far placebo masks from
  conservative neutral reasoning;
- added matched-mask and recovery-validation tests; repository total was 60
  passing tests at closure;
- audited the frozen 2,400-row membership at 1,536 and 2,048 tokens;
- found 98.2%–98.6% pairability with 5.4% core and 6.8% broad masked-token
  shares at 2,048;
- obtained a historical Claude Opus aggregate-design verdict of CONDITIONAL
  PASS and incorporated manipulation, positive-control, placebo-locality,
  dose, loss-mass, and inference-plan requirements;
- stopped external Opus validation at the user's direction;
- froze a 520-row contextual validation sample with seven fixed 60-row gates
  and one 100-row omitted-awareness diagnostic;
- excluded 3,749 training or previously labelled source rows and enforced 520
  unique source rows;
- independently recomputed every selector, context window, placebo
  contribution, hash, chunk partition, and confidence-bound operating rule;
  the freeze audit passed every assertion;
- completed all 520 blinded contextual annotations without Opus;
- verified exact coverage, unique IDs, allowed labels, and all frozen hashes;
- froze the merged annotations at SHA-256
  `7c8aa537e3fcaaae8cc982a64cc68955d95aa9ad739e603da13745769aa19506`;
- scored the preregistered confidence-bound gates without revising any
  annotation after unblinding;
- observed 57/60 awareness successes in the core-Wood stratum, 9/60
  contaminated near placebos, and 3/60 contaminated distant placebos;
- stopped the semantic experiment before calibration or training, as required
  by the frozen protocol; no fallback dose or placebo was substituted;
- recorded the full result in `reports/CP3_RECOVERY_RESULT.md`.

Exit criterion:

- Met as a no-go. Any continuation requires a new estimand, protocol, and
  fresh holdout; the exposed recovery sample cannot be reused as a gate.

## CP3-T0 — Temporal-persistence novelty and estimand gate

Status: conditional — exact decomposition survives; generic benchmark does not

Artifacts:

- `reports/NEXT_BRANCH_RESEARCH.md`
- `TEMPORAL_COMMITMENT_PROTOCOL.md`

Decision:

- do not revive awareness-span masking or same-trace placebo training;
- pursue a gated causal decomposition of post-steering persistence into
  visible-text, hidden-cache, and ongoing-computation channels;
- treat generic "continuous steering is stronger" as insufficient because
  adjacent refusal-agent and cache-control work already makes it plausible;
- require clean-cache replay, forced identical prefixes, random-direction
  schedules, and reversal-order tests for the result to be informative.
- treat finite-versus-continuous steering alone as insufficient given adjacent
  agent-transfer and cache-control work;
- stop and widen the search if local constraints remove the text/cache
  decomposition or reversal test.

Adversarial self-review:

- the strongest prior-art overlap is continuous state injection in a
  refusal-style agent and cache-conditioned decoding;
- the branch remains distinct only through its evaluation-awareness target,
  exact text/cache decomposition, and commitment/reversal estimand;
- a 1.7B result cannot establish the mechanism in the released 49B organism;
- failure of the local organism gate stops the branch.

Validation policy:

- the user ended external Opus validation on 2026-09-11;
- future checkpoints use automated tests plus documented adversarial
  self-review.

## CP3-T1 — Local temporal-study compute benchmark

Status: in progress — conditional hardware result; organism training blocked

Required:

- replace dead-experiment wall-clock assumptions with the exact temporal-study
  workload;
- complete chunked-loss equivalence;
- complete warmed training and generation benchmarks;
- record memory and swap behavior;
- issue a numeric go/no-go before organism training.

Completed:

- implemented and gradient-validated chunked causal-LM loss;
- completed a 50-step, accumulation-8, 1,536-token LoRA benchmark;
- measured 233.18 supervised tokens/s and stable 34.57--36.13 second steps;
- isolated padded-batch corruption to MPS SDPA and permanently excluded that
  decode configuration;
- confirmed eager attention yields coherent padded batches;
- confirmed non-thinking eager decoding produces fenced answers in 16/16
  prompts.

Open blockers:

- measured training and decode windows recorded swap activity;
- thinking mode never completed within 512 generated tokens;
- non-thinking mode had 5/16 truncations at 512 tokens;
- non-thinking changes the template and requires a new token/mask audit;
- a clean-memory batch-size and generation-cap repeat is required.

Artifact:

- `reports/CP3_T1_PROGRESS.md`
