"""Fast, dependency-light tests for the global-metadata baseline firewall."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark_v1/scripts/run_qpu_global_metadata_mlp_baseline_v1.py"
spec = importlib.util.spec_from_file_location("global_metadata_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(baseline)


class FeatureFirewallTests(unittest.TestCase):
    def test_exact_seven_log1p_features_in_contract_order(self):
        row = {
            "active_width": "3", "structural_depth": "7", "one_qubit_count": "15",
            "two_qubit_count": "2", "swap_like_count": "1", "measurement_count": "4",
            "shots": "1024", "source_id": "source_must_not_enter", "backend": "backend_must_not_enter",
            "target_seconds": "should_not_be_parsed",
        }
        actual = baseline.global_log_features(row)
        import math
        self.assertEqual(len(actual), 7)
        self.assertEqual(actual, tuple(math.log1p(float(row[name])) for name in baseline.FEATURE_FIELDS))

    def test_missing_negative_and_nonfinite_global_feature_rejected(self):
        base = {name: "1" for name in baseline.FEATURE_FIELDS}
        for bad in ("", "-1", "nan", "inf"):
            row = dict(base, shots=bad)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                baseline.global_log_features(row)

    def test_sidecar_unavailable_or_non_single_circuit_rejected(self):
        row = {name: "1" for name in baseline.FEATURE_FIELDS}
        row.update(availability_status="available", circuit_count="1")
        canonical = {"a": {}}
        baseline.validate_feature_coverage(canonical, {"a": row})
        row["circuit_count"] = "2"
        with self.assertRaises(ValueError):
            baseline.validate_feature_coverage(canonical, {"a": row})
        row["circuit_count"] = "1"
        row["availability_status"] = "unavailable"
        with self.assertRaises(ValueError):
            baseline.validate_feature_coverage(canonical, {"a": row})


class SplitFirewallTests(unittest.TestCase):
    @staticmethod
    def small_assignment(shared_train_test_group: bool = False):
        canonical = {}
        outer = {}
        for index in range(10):
            fold = index % 5
            cid = f"row{index}"
            canonical[cid] = {"canonical_row_id": cid}
            group = "shared" if shared_train_test_group and index in (0, 1) else f"g{index}"
            outer[cid] = {"canonical_observation_id": cid, "outer_fold": str(fold),
                          "unified_leakage_group_id": group}
        inner = []
        for outer_fold in range(5):
            for index in range(10):
                cid = f"row{index}"
                test = index % 5 == outer_fold
                inner.append({"canonical_observation_id": cid, "outer_fold": str(outer_fold),
                              "inner_fold": "" if test else str(index % 4)})
        return canonical, outer, inner

    def test_partition_enforces_outer_and_inner_group_firewalls(self):
        canonical, outer, inner = self.small_assignment()
        folds = baseline.validate_fold_assignments(canonical, outer, inner)
        self.assertEqual(set(folds), {0, 1, 2, 3, 4})
        self.assertEqual(folds[0]["group_overlap_count"], 0)
        self.assertEqual(set(folds[0]["inner_validation_rows"]), {"0", "1", "2", "3"})

    def test_partition_rejects_outer_group_overlap(self):
        canonical, outer, inner = self.small_assignment(shared_train_test_group=True)
        with self.assertRaisesRegex(ValueError, "leakage groups cross train/test"):
            baseline.validate_fold_assignments(canonical, outer, inner)

    def test_partition_rejects_non_null_inner_assignment_on_test(self):
        canonical, outer, inner = self.small_assignment()
        test_row = next(row for row in inner if row["outer_fold"] == "0" and row["canonical_observation_id"] == "row0")
        test_row["inner_fold"] = "0"
        with self.assertRaisesRegex(ValueError, "test ID has inner fold"):
            baseline.validate_fold_assignments(canonical, outer, inner)

    def test_graph_unavailable_row_is_excluded_from_every_train_but_remains_fold0_test(self):
        canonical, outer, inner = self.small_assignment()
        folds = baseline.validate_fold_assignments(
            canonical, outer, inner, exclude_from_all_training={"row0"}
        )
        self.assertNotIn("row0", folds[0]["train_ids"])
        self.assertIn("row0", folds[0]["test_ids"])
        for fold in range(1, 5):
            self.assertNotIn("row0", folds[fold]["train_ids"])
            self.assertNotIn("row0", folds[fold]["test_ids"])


class LockedFitEnvelopeTests(unittest.TestCase):
    def test_output_envelope_exists_before_trainer_and_trainer_runs_inside_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fold_0"
            state = {"held": False}

            class Lock:
                def __enter__(self):
                    prelock = json.loads((output / "run_manifest.json").read_text())
                    self_outer.assertEqual(prelock["stage"], "waiting_for_compute_lock")
                    self_outer.assertTrue((output / "run_manifest.json").is_file())
                    state["held"] = True

                def __exit__(self, *_):
                    state["held"] = False

            def trainer(args, preflight, out):
                self.assertTrue(state["held"])
                initial = json.loads((out / "run_manifest.json").read_text())
                self.assertEqual(initial["stage"], "compute_lock_acquired")
                self.assertTrue(initial["compute_lock"]["shared_timing_and_host_gpu_lease_held"])
                return {"fit": "fake"}

            self_outer = self
            result = baseline.execute_fit_with_lock(
                SimpleNamespace(output_dir=output, fold=0), {"input_hashes": {"x": "abc"}}, Lock, trainer
            )
            self.assertEqual(result, {"fit": "fake"})
            self.assertFalse(state["held"])

    def test_existing_output_rejected_before_lock_or_trainer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "already_there"
            output.mkdir()

            def forbidden():
                self.fail("lock must not be entered for an existing output directory")

            def trainer(*_):
                self.fail("trainer must not run for an existing output directory")

            with self.assertRaises(FileExistsError):
                baseline.execute_fit_with_lock(
                    SimpleNamespace(output_dir=output, fold=0), {"input_hashes": {}}, forbidden, trainer
                )

    def test_training_exception_leaves_failed_manifest_and_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fold_2"

            class Lock:
                def __enter__(self): pass
                def __exit__(self, *_): pass

            def trainer(*_):
                raise RuntimeError("mock fit failure")

            with self.assertRaisesRegex(RuntimeError, "mock fit failure"):
                baseline.execute_fit_with_lock(
                    SimpleNamespace(output_dir=output, fold=2), {"input_hashes": {}}, Lock, trainer
                )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "mock fit failure")


if __name__ == "__main__":
    unittest.main()
