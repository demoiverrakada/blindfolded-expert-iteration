# CP1 data and target audit

Date: 2026-09-10

## Input provenance

- Dataset: released round-one expert-iteration CSV
- Dataset repository revision:
  `36c9b63baeaf7072ac38a09ec385b688bc23e3b7`
- Rows: 18,435
- Unique system/user prompt pairs: 18,435
- SHA-256:
  `645116309ca3d3fb6dce849790505b64f05cb67d5d1911878a5daafbf2fa0286`
- Runtime: Python 3.9.6
- Audit command is embedded in `reports/cp1_data_audit_v2.json`.

## Evaluation-awareness leakage

### Frozen detector outputs

| Detector | Rows | Fraction | Interpretation |
|---|---:|---:|---|
| V1 phrase detector | 11,660 | 63.25% | High-precision lower bound |
| V2 phrase detector | 14,890 | 80.77% | Higher-recall operational detector |
| Wood Labs literal mention | 9,685 | 52.54% | Independent surface signal |
| Nemotron self-identification | 824 | 4.47% | Narrow self-reference signal |

V1 is retained unchanged for reproducibility. V2 adds plural context nouns,
broader grammatical subjects, and phrases such as "being evaluated by" while
continuing to reject generic coding uses such as "test the function."

### Independent sample validation

Claude Opus independently labeled a frozen V1-stratified sample containing 25
V1-positive and 25 V1-negative full completions.

Inclusive definition:

- V1: TP 25, FP 0, TN 6, FN 19.
- V1 precision: 100%; recall: 56.8%.
- V2 evaluated on the same sample: TP 41, FP 0, TN 6, FN 3.
- V2 precision: 100%; recall: 93.2%.

Because V2 contains every V1 pattern, the 93.2% recall measured on a
V1-stratified sample is upward-biased. A fresh V2-stratified sample produced:

- Inclusive: TP 25, FP 0, TN 15, FN 10; precision 100%, recall 71.4%.
- Strict: TP 24, FP 1, TN 20, FN 5; precision 96.0%, recall 82.8%.

Prevalence reweighted on the V2 strata was 88.5% inclusive and 81.4% strict.
The strict-label rubric is not yet sufficiently operationalized to treat the
difference between strict estimates as inferential.

Strict definition, excluding evaluator-company/professional-style borderline
cases:

- V1: TP 19, FP 6, TN 9, FN 16.
- V2: TP 32, FP 9, TN 6, FN 3.

Because the sample was stratified on V1 output, population prevalence is
estimated by reweighting the positive and negative strata:

- Inclusive estimate: 91.18%.
- Strict estimate: 71.59%.

These are small-sample estimates, not exact census labels. The robust claim is
that 63.25% is a lower bound and direct target-side awareness language is
widespread.

## Where the leakage occurs

| Segment | V2 matches | Fraction |
|---|---:|---:|
| Full completion | 14,890 / 18,435 | 80.77% |
| Reasoning span | 14,835 / 18,435 | 80.47% |
| Raw suffix after `</think>` | 366 / 18,435 | 1.99% |
| Conservatively retained answer-only target | 291 / 18,054 | 1.61% |

Wood Labs appears in 1.48% of retained answer-only targets. Thus answer-only
removes almost all, but not literally all, direct awareness language.

## Conservative answer-only transform

Inclusion requires:

1. exactly one ordered `<think>...</think>` pair;
2. no non-whitespace text before `<think>`;
3. a non-empty answer suffix;
4. at least one balanced fenced code block.

Results:

| Status | Rows |
|---|---:|
| Included | 18,054 |
| Unterminated reasoning block | 128 |
| Multiple reasoning tags | 62 |
| No complete code fence | 158 |
| Odd/unclosed code-fence count | 33 |

The source row index is retained. Exclusions are not repaired heuristically.

Derived artifact:

- Bytes: 22,277,528
- SHA-256:
  `52a28bc40f468c6f5b39e3b8e190e7507785d28a4d488e3d759e5bdc34d73796`

## Exact token audit

Tokenizer:

- Model: `nvidia/Llama-3_3-Nemotron-Super-49B-v1`
- Revision: `387156d8d6868c19f3472fa607aa9bfc4f662333`
- Transformers: 4.55.2
- Maximum training length: 3,500
- Matched rows: 18,054

| Target | Mean target tokens | Median | P95 | Truncated |
|---|---:|---:|---:|---:|
| Full trace | 1,060.01 | 1,040 | 1,646 | 0 |
| Answer only | 148.01 | 137 | 270 | 0 |

At 300 optimizer steps and global batch size 8:

- Full trace: approximately 2.544 million supervised tokens.
- Answer only: approximately 0.355 million supervised tokens.

The 7.16× difference means full-trace versus answer-only is a bundled
intervention involving awareness language, all other reasoning, output format,
and supervision quantity. It is unsuitable as the primary clean factor.

## Design consequence

The primary target factor is changed to:

- `full_loss`: ordinary completion-only loss;
- `awareness_masked_loss`: identical tokens and forward pass, but labels
  corresponding to complete awareness-matching reasoning sentences are set to
  `-100`.

This preserves prompts, token sequences, reasoning context, formatting,
examples, sequence lengths, and hook computation. It tests the effect of
direct token-level supervision of verbalized awareness. Because awareness
tokens remain teacher-forced inputs to later tokens, it does not remove all
awareness information.

Answer-only remains a diagnostic extreme, not a primary factorial arm.

## Validation state

- Eleven local unit tests pass.
- V1 sample labels and Opus disagreements are committed as structured data.
- Opus independently reviewed the V2-stratified 50-row sample.
- CP1 revalidation verdict: PASS.
- The actual V2-or-Wood-Labs masking union is a CP2 validation target.
