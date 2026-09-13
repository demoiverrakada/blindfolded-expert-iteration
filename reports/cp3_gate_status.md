# CP3 detector and mask-validation checkpoint

## Detector revalidation

Status: **FAIL — no-go under the frozen protocol**

The revision implementation audit passed on all 1,213,324 eligible sentences,
and annotation provenance covers all 979 unique blinded sentences exactly.
The detector score failed one preregistered gate:

- V2 pattern 9 precision: 24/30 = 0.80 (required: 0.92)

All other detector gates passed:

- weighted sentence-intrinsic recall: 0.8413 (required: 0.80)
- V2 pattern 12 precision: 28/30 = 0.9333
- added family 1 precision: 30/30 = 1.00
- added family 2 precision: 28/30 = 0.9333
- added family 3 precision: 28/30 = 0.9333
- added-only hits in the 200-row neutral-code holdout: 0

The protocol allowed one detector revision. No additional regex editing or
rescoring is permitted against this validation sample.

## V3 mask validation

Status: **ANNOTATED, ADJUDICATION PENDING**

All 122 blinded rows were annotated with exact ID coverage. Two exact repeated
sentences received inconsistent labels across independent chunks. The scorer
correctly stopped before producing a report. A two-row opaque-ID adjudication
file is prepared, but no labels have been changed.

Even if this adjudication later passes, the detector no-go remains decisive for
the current training protocol.
