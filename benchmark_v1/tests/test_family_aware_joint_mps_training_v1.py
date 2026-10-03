from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import train_family_aware_joint_mps_v1 as trainer  # noqa: E402


def _label(status="runtime_observed", quality=False, runtime=0.25):
    return {"target_status": status,
            "runtime_seconds": runtime if status == "runtime_observed" else None,
            "quality_pass": quality if status == "runtime_observed" else None,
            "minimum_passing_chi": "", "session_fidelity": [0.999] * 3}


def test_minimum_passing_chi_requires_all_six_rungs_technically_complete():
    complete = {("h", chi): _label(quality=(chi >= 8)) for chi in trainer.CHI}
    trainer.attach_minimum_passing_chi(complete, ["h"])
    assert {complete[("h", chi)]["minimum_passing_chi"] for chi in trainer.CHI} == {8}

    partial = {("h", chi): _label(quality=(chi == 2)) for chi in trainer.CHI}
    partial[("h", 64)] = _label(status="unavailable_technical")
    trainer.attach_minimum_passing_chi(partial, ["h"])
    assert {partial[("h", chi)]["minimum_passing_chi"] for chi in trainer.CHI} == {""}

    no_pass = {("h", chi): _label(quality=False) for chi in trainer.CHI}
    trainer.attach_minimum_passing_chi(no_pass, ["h"])
    assert {no_pass[("h", chi)]["minimum_passing_chi"] for chi in trainer.CHI} == {"NO_PASS"}


def test_fold_preparation_crossfits_family_inputs_for_each_registered_seed(monkeypatch):
    calls = []

    def fake_fit(_x, _y, vocab, seed):
        calls.append(seed)
        return {"constant": sorted(vocab)[0]}

    monkeypatch.setattr(trainer, "fit_family_classifier", fake_fit)
    hashes = [f"h{i}" for i in range(10)]
    features = {
        qhash: {
            "c44_fold": str(index % 5),
            "family_holdout_fold_diagnostic_only": str(index % 5),
            "family_component_training_label_only": f"family-{index % 2}",
            "active_width": "4", "measurement_stripped_structural_depth": str(index + 1),
            "one_qubit_gate_count": "5", "two_qubit_gate_count": "3",
            "multi_qubit_gate_count": "0", "swap_like_gate_count": "0",
        }
        for index, qhash in enumerate(hashes)
    }
    labels = {(qhash, chi): _label(quality=(chi >= 4)) for qhash in hashes for chi in trainer.CHI}
    data = {"feature_by_hash": features, "labels": labels}

    prepared = trainer.prepare_fold(data, "c44", 0)

    assert set(prepared["train_hashes"]).isdisjoint(prepared["test_hashes"])
    assert set(prepared["train_family_by_seed"]) == set(trainer.SEEDS)
    assert set(prepared["test_family_prediction_by_seed"]) == set(trainer.SEEDS)
    assert all(values.shape == (len(prepared["train_cells"]),)
               for values in prepared["train_family_by_seed"].values())
    assert all(set(predictions) == set(prepared["test_hashes"])
               for predictions in prepared["test_family_prediction_by_seed"].values())
    assert set(calls) == set(trainer.SEEDS)
