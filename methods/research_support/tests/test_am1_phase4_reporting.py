"""Reject count rounding in the saved-evidence-only reporting correction."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_am1_phase4 import integral_count


def test_exact_integral_decimals():
    assert integral_count("341.0") == "341"
    assert integral_count("341") == "341"
    assert integral_count("0.000") == "0"


def test_no_rounding():
    with pytest.raises(ValueError):
        integral_count("341.01")


def test_invalid_counts():
    for value in ("NaN", "Infinity", "-1"):
        with pytest.raises(ValueError):
            integral_count(value)
