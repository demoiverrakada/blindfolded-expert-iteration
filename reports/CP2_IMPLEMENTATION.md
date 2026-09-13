# CP2 awareness-mask and projection implementation

Date: 2026-09-10

## Operational awareness mask

The mask is applied only to complete sentences inside the assistant
`<think>...</think>` span.

A sentence is selected when it:

- matches frozen detector V2; or
- contains a literal Wood Labs / woodlabs mention.

Final-answer tokens are never selected by this mask.

The detector definitions are centralized in `src/bei/detectors.py` and imported
by both the CP1 audit and CP2 training preprocessing. This prevents audit/train
pattern drift.

## Full-dataset mask audit

Dataset rows: 18,435  
Tokenizer revision:
`nvidia/Llama-3_3-Nemotron-Super-49B-v1@387156d8d6868c19f3472fa607aa9bfc4f662333`

| Quantity | Result |
|---|---:|
| Included well-formed rows | 18,245 |
| Excluded malformed reasoning rows | 190 |
| Rows with at least one masked token | 15,535 (85.15%) |
| Total supervised completion tokens | 19,456,396 |
| Masked completion tokens | 1,497,086 |
| Masked share of supervised tokens | 7.69% |
| Mean masked tokens per row | 82.05 |
| Median masked tokens per row | 54 |
| P95 masked tokens per row | 244 |
| Mean awareness spans per row | 3.40 |

The treatment is therefore neither vanishingly sparse nor equivalent to
removing most of the training signal.

A 25-masked/25-unmasked sentence-level sample with prompts and local context is
frozen at `reports/cp2_sentence_validation_sample.csv` for Claude review.

The corrected audit also reports:

- zero prefix-tokenization mismatches;
- zero awareness spans without an overlapping token;
- per-context masked shares from 3.47% to 15.10%;
- median per-row masked share 5.20%, P95 27.34%, maximum 68.76%.

## Token-label alignment

`mask_labels_for_spans`:

- requires labels and tokenizer offsets to have equal length;
- ignores special tokens with `(0, 0)` offsets;
- masks only tokens overlapping declared character spans;
- accepts a minimum token index so prompt labels remain independently masked;
- fails closed on invalid spans.

The full audit runs the exact Nemotron tokenizer and the upstream chat-template
construction over every row before counting masked labels.

## Projection implementation

`src/bei/projection.py` provides:

- column orthonormalization with rank and finite-value checks;
- deterministic seeded random rank-matched controls;
- projection of tensor and tuple block outputs;
- activation dtype and device preservation;
- explicit hidden-dimension validation, which catches feature-shard mismatch;
- post-block forward hooks at declared decoder layers;
- a zero-scale identity condition that executes the same matrix-multiply path;
- context-managed hook cleanup.

For activation `h` and orthonormal basis `Q`, the hook computes:

`h' = h - scale * Q(Q^T h)`.

## Local verification

Thirty-five unit tests pass under:

- Python 3.9.6
- PyTorch 2.7.1
- Transformers 4.55.2

Tests cover:

- frozen detector nesting and generic-test rejection;
- complete-sentence selection and final-answer exclusion;
- character-offset/token-label overlap;
- special-token handling;
- answer extraction and token-budget arithmetic;
- orthonormalization and rank-deficiency rejection;
- exact removal of basis components;
- projected gradients;
- tuple-output preservation;
- dtype preservation;
- deterministic random controls;
- identity scale;
- hook installation and cleanup.

## Deliberate non-claims

- Sentence masking removes direct loss on detected awareness language; it does
  not remove those teacher-forced tokens as inputs to later predictions.
- Local toy tests do not establish compatibility with the real 49B PEFT model.
- DDP replicates the model rather than sharding the hidden dimension in the
  released setup, but the real remote smoke test must still assert residual and
  basis dimensions on every process.
- The mask union's sentence-level semantic precision and recall are pending
  Claude revalidation.

## First Claude validation

Verdict: FAIL.

Blocking findings fixed before revalidation:

- the rubric now explicitly includes bare Wood Labs mentions;
- 190 malformed reasoning rows are excluded identically from all primary arms;
- hook layer indices are validated before any hook is registered;
- projection accumulation uses detached fp32 bases and fp32 arithmetic;
- sentence-level rather than row-level validation data is emitted;
- every span must overlap a token and prefix tokenization is asserted;
- per-context and per-row masked fractions are reported.

## Final Claude revalidation

Verdict: PASS.

On the sentence-level sample:

- TP: 25
- FP: 0
- TN: 25
- FN: 0
- disagreement source indices: none

This supports mask precision. It does not strongly identify recall because
ordinary coding fragments dominate the unmasked-sentence population.

CP3 must carry:

- an awareness-enriched sample for recall;
- targeted precision checks for broad V2 patterns 9 and 12;
- fail-closed zero-token-span behavior in the training collator;
- real Nemotron/PEFT hook and fp32-projection validation.
