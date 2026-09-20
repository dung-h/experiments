#!/usr/bin/env python3
"""Create a portable canonical corpus from compatible dense-statevector runs.

Only runs with the same target semantics and repeat protocol may be combined.
The command validates that contract, reconstructs pre-execution graph features
for old v1 rows and preserves a source-run identifier for every record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulator_runtime_v1"))
import run_torch_statevector_matrix as v1  # noqa: E402
from features import static_features  # noqa: E402
from run_dense_statevector_v2 import build_spec  # noqa: E402


TARGET = "warm_execution_median_seconds"


def feature_row(row: pd.Series) -> dict[str, object]:
    spec = build_spec(
        str(row["family"]), int(row["num_qubits"]), int(row["depth_parameter"]), int(row["seed"])
    )
    if spec.circuit_id != str(row["circuit_id"]):
        raise ValueError(f"circuit hash mismatch for {row['record_id']}")
    return static_features(spec, str(row["precision"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-csv", type=Path, required=True)
    parser.add_argument("--v2-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = (("v1_reference", args.v1_csv), ("v2_structural", args.v2_csv))
    frames: list[pd.DataFrame] = []
    for source_name, path in sources:
        raw = pd.read_csv(path)
        if len(raw) == 0:
            raise ValueError(f"empty input: {path}")
        required = {"record_id", "circuit_id", "family", "num_qubits", "depth_parameter", "seed", "precision", "status", "repeat_count", "warmup_count", TARGET}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        if set(raw["repeat_count"].dropna().astype(int)) != {3} or set(raw["warmup_count"].dropna().astype(int)) != {1}:
            raise ValueError(f"{path} does not use the compatible 1-warmup/3-repeat protocol")
        raw = raw.copy()
        raw["source_run"] = source_name
        frames.append(raw)

    corpus = pd.concat(frames, ignore_index=True, sort=False)
    if corpus["record_id"].duplicated().any():
        duplicates = corpus.loc[corpus["record_id"].duplicated(), "record_id"].head(3).tolist()
        raise ValueError(f"duplicate record_id across source runs: {duplicates}")
    feature_rows = [feature_row(row) for _, row in corpus.iterrows()]
    features = pd.DataFrame(feature_rows)
    for column in features:
        if column in corpus and not corpus[column].isna().all():
            observed = corpus[column].notna()
            if pd.api.types.is_numeric_dtype(features[column]):
                mismatch = observed & ~np.isclose(
                    pd.to_numeric(corpus[column], errors="raise"),
                    pd.to_numeric(features[column], errors="raise"),
                )
            else:
                mismatch = observed & (corpus[column].astype(str) != features[column].astype(str))
            if mismatch.any():
                raise ValueError(f"feature mismatch in {column}")
        corpus[column] = features[column]
    corpus["corpus_record_id"] = corpus["source_run"] + ":" + corpus["record_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.output, index=False)
    summary = {
        "rows": int(len(corpus)),
        "successful_rows": int((corpus["status"] == "ok").sum()),
        "source_runs": corpus.groupby("source_run").size().to_dict(),
        "families": sorted(corpus["family"].unique().tolist()),
        "widths": sorted(int(value) for value in corpus["num_qubits"].unique()),
        "contexts": sorted(corpus["context_id"].unique().tolist()),
        "target": TARGET,
        "protocol": "same prepared_execute_reset_state target; one warm-up and three warm repeats",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
