# CP2 Claude Opus validation

Date: 2026-09-10

## First implementation review

Verdict: FAIL.

Blocking findings:

- rubric and bare-Wood-Labs implementation differed;
- malformed reasoning traces were handled inconsistently;
- partial hook installation could leak after an exception;
- validation was row-level rather than sentence-level;
- span/token and prefix-boundary assertions were incomplete;
- context-specific mask imbalance was hidden.

All six blockers were fixed and committed.

## Final revalidation

Commit:
`d9189294979ae238f725d7fa6032cc689e5794a9`

Verdict: PASS.

Independent sentence labels:

| Result | Count |
|---|---:|
| TP | 25 |
| FP | 0 |
| TN | 25 |
| FN | 0 |

No disagreement source indices.

Opus confirmed:

- rubric/code agreement;
- symmetric malformed-row exclusion;
- atomic hook installation;
- sentence-level validation;
- zero prefix mismatches and zero zero-token spans;
- per-context and per-row mask reporting;
- fp32 detached projection math and cleanup.

Carry-forward limitations:

- the random unmasked sample gives weak recall information;
- V2 patterns 9 and 12 require targeted precision checks;
- real Nemotron/PEFT integration remains untested;
- training preprocessing must fail closed if any span maps to zero tokens.
