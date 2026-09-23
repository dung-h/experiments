# Azizov-style compiled pretrain → QPU transfer

Finding date: 2026-09-22.

This experiment asks whether a compiled-feature simulator model can be
transferred onto Ma–Li's 340 Osaka/Kyoto `result.time_taken` labels.

It is not Ma–Li's source-DAG Graph Transformer, and it does not re-measure
Azizov's Aer `T_exec` table. Simulator labels remain the public
Washington/Sherbrooke `time_taken` CSVs. Compile targets are current
FakeWashingtonV2/FakeSherbrooke snapshots. QPU inputs are current
FakeOsaka/FakeKyoto compiled features.

Stage 1: transpile every Ma–Li QASM to Washington and Sherbrooke, extract
compiled T1/T2 features, predict simulator `time_taken`.

Stage 2: keep that Ridge mapping, fit an affine head on QPU train folds, and
score the 340 hardware labels. Scratch compiled QPU models are the ceiling.
Same-circuit `incl` matches Ma–Li's transfer setting. `excl` removes the test
QASM from simulator pretraining. Family hold-out is the B2 analogue with
matched simulator compilation.
