> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Static validation report

Status: **PASS — documentation/asset feasibility only**. This is not a timing,
inference, scientific result or public-release PASS.

- All ten requested artifacts present; one provenance manifest; one stdlib
  documentation builder/validator, not an AM1 timing harness.
- Inventory: 86 distinct track/context/method rows, 23 CSV columns including
  fidelity card/class; exact 34 QPU and 49 simulator result rows covered, plus original
  Scholten/nativeTN/analog companion. Status totals reproduce README/table.
- 30 parent QPU + 60
  dense + 60 joint runtime +
  30 family classifier checkpoint
  SHA256s match producing receipts; no state deserialization or model imports.
- 25 saved transform files pinned; seven complete scalar references verified
  single-valued per fold/fold-source and preserved in provenance. Maestro 8 width
  knots contain saved finite coefficients. Five unsaved-model roots audited;
  dense classical states also absent per producing code/output inventory;
  Hyb Ridge receipts contain selection info but no coefficients/scalers.
- Saved closure validation rerun; 30 loop timer boundaries and 603-session TN
  telemetry hierarchy reproduced, E2E nonempty0. No source/history rewrite.
- Every direct source/asset digest is recorded in manifest. Default validator
  checks source digests, exact payload bytes, inventory coverage/uniqueness,
  local Markdown links and status counts. Experimental-directory absence and
  planned-vs-measured wording are creation-time scope checks. Source digests rerun
  after report creation; assertion failures stop validation.
- Environment versions were read via distribution metadata, never Torch CUDA
  import or availability calls. GPU/host inventory was read-only nvidia-smi,
  lscpu/free/df; volatile inventory is not an estimator measurement.

Revalidate (reads only; no model/checkpoint loading):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/build_am1_feasibility.py
```

Pins: provenance_manifest.json (historical source; not bundled). The manifest excludes
itself from payload hashes; no self-hash cycle. It pins this static builder.
No training/refit/inference/QPU/simulator/QCEC/dependency install/latency timing,
frozen data/protocol/prediction/checkpoint changes, commit/push or presentation
updates. Scientific Freeze v2 preserved. READY remains a static judgment pending
future separately authorized execution/parity tests.
