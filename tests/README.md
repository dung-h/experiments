# Public tests and their limits

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/verify.py
python3 -B scripts/check_qpu_results.py
python3 -B scripts/check_simulator_results.py
python3 -B scripts/check_completed_science.py
python3 -B -m unittest discover -s tests
```

The public suite currently has 36 tests. It checks inventory/path/hash safety,
QPU population and aggregates, retained simulator OOF scores, AM1 Dense
prediction/timing summaries and complete-grid cases/seeds. Altered synthetic
fixtures test rejection behavior; they are not new quantum experiments.

These checks do not establish independent retraining, raw QPU score reproduction,
fresh timing, QCEC execution, universal method fidelity or scheduler utility.
Saved-prediction recomputation differs from inference and data recollection.

[Archived research tests](../methods/research_support/README.md) require private
fixtures/historical environments and are not part of the public suite.
Their source-file count is not a count of passing public tests.
[Provenance](../provenance/README.md) separates historical receipts from fresh checks.
