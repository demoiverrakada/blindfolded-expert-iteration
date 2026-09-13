# CP3-L0 progress

Date: 2026-09-10

Completed:

- isolated Python 3.11 pilot environment and hashed 52-package lock;
- frozen Qwen snapshot and weight-shard hashes;
- MPS access confirmed outside the filesystem sandbox;
- 42 repository tests passed in the pilot environment;
- real bf16 MPS model load and deterministic generation;
- real PEFT rank-16 wrapping over all seven target module types;
- all 28 layers resolved before and after PEFT;
- all 392 trainable LoRA tensors received gradients;
- CPU-fp32 basis factorization with MPS projection;
- scale-zero exact identity and scale-one causal logit change.
- one-directional provenance audit over all 1,140 pinned BigCodeBench tasks,
  18,435 E1 rows, and 311,083 SDF documents;
- 30/30 planted overlap controls recovered and rolling-shingle results checked
  against a brute-force implementation;
- three mutually disjoint 200-task panels frozen with independently rechecked
  SHA-256 hashes;
- one context-stratified 2,400-row E1 membership set frozen for every arm and
  seed;
- no panel task reached the 0.30 quarantine threshold; the maximum observed
  task-side overlap was 0.24.
- full Qwen tokenizer/mask audit over all 17,100 eligible E1 rows;
- exact prefix alignment, supervised `<think>` delimiters, and non-empty token
  overlap for every declared awareness span;
- 1,024-token cap failed; 1,536 and 2,048 passed retention, masked-share, and
  context-composition gates;
- frozen membership retains 1,945 rows at 1,536 and 2,390 rows at 2,048.
- completion-token and post-mask supervised-token budgets use reduced
  no-repetition step counts: 243 at 1,536 and 298 at 2,048;
- Claude Opus tokenizer/mask review: PASS after substitution-context,
  identifier-safety, membership-composition, and artifact-version controls
  were added.

Important preliminary observation:

- cold single-sequence decode was 2.80 tokens/second and 32 tokens did not
  close the thinking block. This is not the preregistered warmed batch-16 gate,
  but it makes decode feasibility the leading risk.

Still required before CP3-L0 can close:

- CP2 enriched recall and broad-pattern precision checks;
- chunked cross-entropy equivalence;
- 50-step training matrix at 1,024, 1,536, and 2,048;
- warmed batch-16, 512-token decode benchmark;
- hard-stop decision report;
- Claude Opus checkpoint review: PASS after two conditional reviews identified
  and closed missing detector-path controls, a misleading row-screen claim,
  short-solution overlap handling, curly-apostrophe self-identifications, and
  artifact identity checks.
