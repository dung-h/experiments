#!/usr/bin/env python3
"""Post-execution reporting type correction, without changing pinned producers.

The original analyzer assumes count metadata are integer-formatted strings.
The frozen selection also contains exact integral decimal strings ('341.0').
Normalize these strings in an in-memory copy only. No numerical rounding,
selection, timing replay, protocol rewrite or model changes are permitted.
"""
from __future__ import annotations

import argparse
from decimal import Decimal

import analyze_am1_phase4 as original

READ = original.read


def integral_count(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value() or number < 0:
        raise ValueError("nonintegral_count_metadata:" + str(value))
    return str(int(number))


def reporting_read(path):
    value = READ(path)
    if path == original.OUT / "predeclaration.json":
        for row in value["selected"]:
            for key in ("width", "depth", "gates"):
                row[key] = integral_count(row[key])
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("analyze", "validate"), required=True)
    args = p.parse_args()
    original.read = reporting_read
    if args.stage == "analyze":
        original.analyze()
    else:
        print(original.validate())


if __name__ == "__main__":
    main()
