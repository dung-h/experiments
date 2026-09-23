# Ma--Li QWalk tail forensics

QWalk has two observed rows: one Osaka and one Kyoto measurement of the
same raw-QASM hash. It is therefore retained as a stress case, not scored
as a standalone R² or treated as an iid circuit family.

`qwalk_rows.csv` records the nearest training row in standardized compact
physical-feature space and all available out-of-fold predictions.

`qwalk_feature_comparison.csv` gives train p05/median/p95, percentile and
standardized log-distance for the features most relevant to the tail.

A high percentile or nearest-neighbour distance indicates that the model
is extrapolating. Lower error from `rich_summary` or the ordered-DAG pilot
would show feature coverage helps, but would not prove topology causality.
If QWalk remains far outside train support, the correct remedy is explicit
OOD uncertainty/coverage handling or more representative training data,
not silently pooling the two rows into the headline metric.
