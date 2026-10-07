from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simulator_results", ROOT / "scripts/check_simulator_results.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class SimulatorResultsTests(unittest.TestCase):
    def test_metrics_hand_calculation(self):
        values = CHECK.metrics([(1, 2), (3, 2)])
        self.assertEqual(values["metric_n"], 2)
        self.assertEqual(values["mae_seconds"], 1)
        self.assertEqual(values["r2_seconds"], 0)
        self.assertEqual(values["max_absolute_error_seconds"], 1)

    def test_tail_percentile_uses_linear_interpolation(self):
        self.assertAlmostEqual(CHECK.percentile([0, 10], 0.9), 9)

    def test_constant_target_r2_is_undefined(self):
        self.assertIsNone(CHECK.metrics([(2, 2), (2, 2)])["r2_seconds"])
        self.assertIsNone(CHECK.metrics([(2, 1), (2, 3)])["r2_seconds"])

    def test_single_target_r2_is_undefined(self):
        self.assertIsNone(CHECK.metrics([(2, 2)])["r2_seconds"])

    def test_invalid_values_rejected(self):
        for value in ("nan", "inf", -1, ""):
            with self.assertRaises(ValueError):
                CHECK.number(value)

    def test_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            CHECK.unique_rows([{"hash": "a"}, {"hash": "a"}], ("hash",))

    def test_seed_median_not_mean(self):
        row = {"s1": 1, "s2": 2, "s3": 99, "prediction": 2}
        CHECK.seed_median(row, "prediction", ["s1", "s2", "s3"])
        row["prediction"] = 34
        with self.assertRaises(ValueError):
            CHECK.seed_median(row, "prediction", ["s1", "s2", "s3"])

    def test_aliases_are_scored_once(self):
        a = {"panel_member_id": "a", "source_sha256": "hash", "fold": "0",
             "status": "predicted", "target_seconds": "2", "prediction_seconds": "3"}
        b = dict(a, panel_member_id="b")
        self.assertEqual(CHECK.collapse_aliases([a, b]), [(2, 3)])
        b["fold"] = "1"
        with self.assertRaisesRegex(ValueError, "inconsistent exact-hash aliases"):
            CHECK.collapse_aliases([a, b])

    def test_saved_results(self):
        self.assertEqual(CHECK.Audit(ROOT).run(), (49, 18, 603))


if __name__ == "__main__":
    unittest.main()
