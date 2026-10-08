from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qpu_results", ROOT / "scripts/check_qpu_results.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class QpuResultsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {r["method_id"]: r for r in CHECK.rows(ROOT / "results/real_qpu/method_comparison.csv")}
        self.pair = CHECK.rows(ROOT / "results/real_qpu/shared_row_comparisons.csv")[0]

    def test_published_aggregates(self):
        self.assertEqual(CHECK.check(ROOT), {"variants": 32, "paired_comparisons": 465, "seed_rows": 24})

    def test_wrong_denominator_rejected(self):
        row = copy.deepcopy(self.metrics["mali_logical_graph"])
        row["assigned_rows"] = "4514"
        with self.assertRaisesRegex(ValueError, "counts"):
            CHECK.check_metric(row)

    def test_wrong_coverage_rejected(self):
        row = copy.deepcopy(self.metrics["mali_logical_graph"])
        row["coverage"] = "0.5"
        with self.assertRaisesRegex(ValueError, "coverage"):
            CHECK.check_metric(row)

    def test_unavailable_score_rejected(self):
        row = copy.deepcopy(self.metrics["scholten_original_paper"])
        row["mae_seconds"] = "0"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            CHECK.check_metric(row)

    def test_wrong_observed_delta_rejected(self):
        pair = copy.deepcopy(self.pair)
        pair["observed_mae_delta_seconds"] = "1"
        with self.assertRaisesRegex(ValueError, "paired delta"):
            CHECK.check_pair(pair, self.metrics)

    def test_reversed_interval_rejected(self):
        pair = copy.deepcopy(self.pair)
        pair["bootstrap_ci_low_seconds"] = "2"
        pair["bootstrap_ci_high_seconds"] = "1"
        with self.assertRaisesRegex(ValueError, "reversed"):
            CHECK.check_pair(pair, self.metrics)

    def test_bootstrap_mean_is_not_observed_delta(self):
        pair = copy.deepcopy(self.pair)
        pair["bootstrap_mean_delta_seconds"] = "0.2"
        CHECK.check_pair(pair, self.metrics)


if __name__ == "__main__":
    unittest.main()
