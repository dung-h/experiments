> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Integral count metadata in reporting

Measurement and the saved-evidence validation completed. The first analysis
invocation failed before producing results because the pinned analyzer used
`int("341.0")` for count metadata from the frozen selection.

The producer, original analyzer, predeclaration, source pins, raw measurements,
predictions and inputs remain unchanged. `report_am1_phase4.py` delegates to the
pinned analyzer after normalizing width/depth/gate metadata in an in-memory
copy. Decimal values must be finite, nonnegative and exactly integral. It does
not round a fractional value, modify stored metadata, select observations or
change scientific calculations. Separate regression tests cover this condition.

No inference or timing is repeated for this reporting-only correction.
