# CP1 Claude Opus validation

Date: 2026-09-10  
Reviewer: Claude Opus, high effort, terminal read-only review

## First review

Verdict: FAIL.

Key findings:

- V1 had 100% sampled precision but only 56.8% recall.
- The 63.25% detector output was a lower bound, not the population estimate.
- Answer-only targets retained small residual leakage.
- The original code-fence check accepted incomplete answers.
- Full-trace and answer-only targets differed by an unmeasured token budget.
- Version control, immutable revisions, launch commands, and tests were
  incomplete.

All blocking findings were addressed.

## Revalidation

Commit reviewed:
`2911895717cf8f3ce0ed387228494f1867e0b7bb`

Verdict: PASS.

Fresh V2-stratified sample:

| Definition | TP | FP | TN | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Inclusive | 25 | 0 | 15 | 10 | 100.0% | 71.4% |
| Strict | 24 | 1 | 20 | 5 | 96.0% | 82.8% |

The earlier 93.2% V2 recall estimate was correctly identified as upward-biased
because it was measured on a V1-stratified sample and V2 is a superset of V1.

Opus confirmed that CP1 adequately resolved:

- detector versioning;
- dataset and model provenance;
- answer-segment leakage;
- conservative extraction statuses;
- exact Nemotron token counts;
- the 7.16× answer-only token confound;
- the shift to identical-sequence awareness-loss masking;
- local tests and scientific scope.

## CP2 carry-forward items

- Validate the actual V2-or-Wood-Labs masking union.
- Measure masked-token share before GPU use.
- Operationalize the inclusive span rubric.
- Add tests for label-mask alignment and token-budget arithmetic.
- Validate projection gradients, tuple outputs, dtypes, devices, and hook
  location.
