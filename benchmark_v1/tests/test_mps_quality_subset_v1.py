import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_mps_quality_subset_v1.py"
SPEC = importlib.util.spec_from_file_location("aggregate_mps_quality_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_derived_bootstrap_seed_is_stable_and_distinguishes_comparisons():
    first = MODULE.derived_seed(20260925, "20260925-bootstrap", "artifact", "method-a-minus-ridge")
    second = MODULE.derived_seed(20260925, "20260925-bootstrap", "artifact", "method-a-minus-ridge")
    other = MODULE.derived_seed(20260925, "20260925-bootstrap", "artifact", "method-b-minus-ridge")
    assert first == second
    assert first != other
    assert 0 <= first < 2**32


def test_metric_bundle_uses_linear_interpolated_quantiles():
    result = MODULE.metric_bundle([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0])
    assert result["n"] == 4
    assert result["mae_seconds"] == 1.5
    assert result["p90_abs_error_seconds"] == 2.7
    assert result["max_abs_error_seconds"] == 3.0


def test_output_reports_seeded_paired_bootstrap_point_and_interval():
    result = MODULE.paired_mae_bootstrap([1.0, 2.0], [0.0, 1.0], [1.0, 2.0], seed=5, replicates=100)
    assert result["observed_mae_difference_seconds_left_minus_right"] == 1.0
    assert "bootstrap_mean_difference_seconds" in result
    assert result["replicates"] == 100
    assert len(result["percentile_95_ci_seconds"]) == 2
    assert result["quantile_method"] == "linear interpolation at position (n-1)*p"
