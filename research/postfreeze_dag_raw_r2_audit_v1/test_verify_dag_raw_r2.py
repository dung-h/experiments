#!/usr/bin/env python3
"""Small deterministic unit tests for the read-only DAG metric verifier."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dag_verify", HERE / "verify_dag_raw_r2.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MetricTests(unittest.TestCase):
    def test_raw_and_log_r2_are_computed_in_their_own_scales(self) -> None:
        y = np.asarray([1.0, 2.0, 4.0])
        p = np.asarray([1.0, 3.0, 3.0])
        got = MODULE.metrics(y, p)
        # SSE=2; SST=sum((y-mean(y))^2)=14/3.
        self.assertAlmostEqual(got["r2_raw_seconds"], 1 - 2 / (14 / 3))
        ly, lp = np.log1p(y), np.log1p(p)
        self.assertAlmostEqual(got["r2_log1p_seconds"], 1 - np.sum((lp - ly) ** 2) / np.sum((ly - ly.mean()) ** 2))

    def test_negative_predictions_are_rejected_before_log1p(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, -0.1]))


if __name__ == "__main__":
    unittest.main()
