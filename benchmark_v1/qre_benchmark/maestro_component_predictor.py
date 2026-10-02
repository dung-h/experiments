"""Fold-local component cost surfaces for the Maestro CPU adaptation.

This module predicts one candidate's warm QCSim runtime.  It does not select a
candidate or calculate an oracle.  Coefficients are reconstructed only from
the frozen synthetic calibration allocation.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


class PredictionUnavailable(ValueError):
    """A calibration or circuit feature is outside the declared model domain."""


CALIBRATION_CLASSES = (
    "one_qubit_noncommuting",
    "two_qubit_wrapper_control",
    "two_qubit_cx_interleaved",
)
PARTIAL_CALIBRATION_MANIFEST_ID = "maestro-common-panel-completion-v3-partial-resume"
PARTIAL_CALIBRATION_FIT_VERSION = "paired_endpoint_heldout_v3_partial_resume"


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionUnavailable(f"missing_or_non_numeric:{label}") from exc
    if not math.isfinite(parsed):
        raise PredictionUnavailable(f"non_finite:{label}")
    return parsed


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PredictionUnavailable("empty_calibration_cell")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _slope_through_origin(samples: list[tuple[int, float]]) -> float:
    if len(samples) < 2 or len({x for x, _ in samples}) < 2:
        raise PredictionUnavailable("component_needs_two_repeat_levels")
    numerator = sum(float(x) * y for x, y in samples)
    denominator = sum(float(x) * float(x) for x, _ in samples)
    return numerator / denominator


def _affine_fit(samples: list[tuple[int, float]]) -> tuple[float, float]:
    if len(samples) < 3 or len({x for x, _ in samples}) < 3:
        raise PredictionUnavailable("sampling_needs_three_shot_levels")
    xs = [float(x) for x, _ in samples]
    ys = [y for _, y in samples]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator == 0:
        raise PredictionUnavailable("sampling_shots_not_identifiable")
    per_shot = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    intercept = ybar - per_shot * xbar
    return intercept, per_shot


def _cell_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row.get(name, "") for name in (
        "kind", "candidate", "width", "chi", "operation_class", "shots", "operation_repeats"
    ))


@dataclass(frozen=True)
class FittedCalibration:
    """Knot tables recovered from one immutable calibration allocation."""

    operation_knots: Mapping[tuple[str, str, int, int | None], float]
    sampling_intercept_knots: Mapping[tuple[str, int, int | None], float]
    sampling_per_shot_knots: Mapping[tuple[str, int, int | None], float]
    calibration_raw_sha256: str
    calibration_manifest_id: str
    clock_id: str
    calibration_warm_observations: int
    # V2 keeps invalid knots explicit and distinguishes process-isolated
    # timed observations from the legacy persistent-process warm-row field.
    calibration_timed_observations: int = 0
    # Keys are (candidate, component, width, configured chi). Values are
    # explicit unavailable reasons for expected knots omitted from the numeric
    # coefficient tables.
    invalid_knot_states: Mapping[tuple[str, str, int, int | None], str] = field(default_factory=dict)
    fit_version: str = "through_origin_v1"
    interpolation_width_grid: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    interpolation_chi_grid: tuple[int, ...] = ()
    fit_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    resolved_contract_sha256: str = ""
    run_identity_sha256: str = ""


def fit_frozen_calibration(
    raw_path: Path,
    manifest: Mapping[str, Any],
    clock_id: str = "reported_time_seconds",
) -> FittedCalibration:
    """Reconstruct coefficients from the frozen v3 synthetic calibration rows.

    A caller invokes this separately inside each evaluation outer fold.  The
    same disjoint calibration corpus is used in every fold; no candidate test
    runtime or quality label is accepted by this function.
    """
    raw_bytes = raw_path.read_bytes()
    rows = list(csv.DictReader(raw_bytes.decode("utf-8").splitlines()))
    corpus = manifest["synthetic_corpus"]
    measure = manifest["measurement"]
    candidates = manifest["implementation"]["candidates"]
    repeats = [int(value) for value in corpus["operation_repeat_counts"]]
    shots_grid = [int(value) for value in corpus["sampling_shots"]]
    operation_shots = int(corpus["operation_shots"])
    expected_per_cell = int(measure["sessions"]) * int(measure["timed_warm_repetitions_per_session"])

    values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("stage") != "warm_execute":
            continue
        if row.get("status") != "ok" or row.get(clock_id) in (None, ""):
            continue
        value = _finite(row[clock_id], f"raw.{clock_id}")
        if value < 0:
            raise PredictionUnavailable("negative_calibration_runtime")
        values[_cell_key(row)].append(value)

    medians: dict[tuple[str, ...], float] = {}
    for key, cell_values in values.items():
        if len(cell_values) != expected_per_cell:
            raise PredictionUnavailable(
                f"incomplete_calibration_cell:{'/'.join(key)}:{len(cell_values)}"
            )
        medians[key] = _median(cell_values)

    operation_knots: dict[tuple[str, str, int, int | None], float] = {}
    sampling_intercept_knots: dict[tuple[str, int, int | None], float] = {}
    sampling_per_shot_knots: dict[tuple[str, int, int | None], float] = {}

    def cell(kind: str, candidate: str, width: int, chi: int | None,
             op: str, shots: int, repeat: int) -> float:
        key = (kind, candidate, str(width), "" if chi is None else str(chi), op,
               str(shots), str(repeat))
        try:
            return medians[key]
        except KeyError as exc:
            raise PredictionUnavailable(f"missing_calibration_cell:{'/'.join(key)}") from exc

    for candidate in candidates:
        widths = corpus["statevector_widths"] if candidate == "statevector" else corpus["mps_widths"]
        bonds = [None] if candidate == "statevector" else corpus["mps_configured_bonds"]
        for width_value in widths:
            width = int(width_value)
            for chi_value in bonds:
                chi = None if chi_value is None else int(chi_value)
                sample_points = [(shots, cell("sampling", candidate, width, chi,
                                               "zero_gate_sample", shots, 0))
                                 for shots in shots_grid]
                intercept, per_shot = _affine_fit(sample_points)
                key = (candidate, width, chi)
                sampling_intercept_knots[key] = intercept
                sampling_per_shot_knots[key] = per_shot

                word_slopes: dict[str, float] = {}
                for op in CALIBRATION_CLASSES:
                    word_slopes[op] = _slope_through_origin([
                        (repeat, cell("operation", candidate, width, chi, op,
                                      operation_shots, repeat))
                        for repeat in repeats
                    ])

                normalizer_1q = (2 ** width) if candidate == "statevector" else int(chi) ** 2
                normalizer_cx = (2 ** width) if candidate == "statevector" else width * int(chi) ** 3
                one_q = word_slopes["one_qubit_noncommuting"] / (2.0 * normalizer_1q)
                wrapper = word_slopes["two_qubit_wrapper_control"]
                cx_composite = word_slopes["two_qubit_cx_interleaved"]
                cx = (cx_composite - wrapper) / normalizer_cx
                if one_q < 0:
                    raise PredictionUnavailable(f"negative_one_qubit_coefficient:{candidate}:{width}:{chi}")
                if cx < 0:
                    raise PredictionUnavailable(f"negative_cx_differential:{candidate}:{width}:{chi}")
                operation_knots[(candidate, "one_qubit", width, chi)] = one_q
                operation_knots[(candidate, "two_qubit_cx", width, chi)] = cx

    return FittedCalibration(
        operation_knots=operation_knots,
        sampling_intercept_knots=sampling_intercept_knots,
        sampling_per_shot_knots=sampling_per_shot_knots,
        calibration_raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        calibration_manifest_id=str(manifest["manifest_id"]),
        clock_id=clock_id,
        calibration_warm_observations=sum(len(values) for values in values.values()),
    )


def _pchip_derivatives(xs: list[float], ys: list[float]) -> list[float]:
    """PCHIP derivatives using the Fritsch-Butland weighted harmonic mean."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise PredictionUnavailable("pchip_needs_two_matching_knots")
    h = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if any(delta <= 0 for delta in h):
        raise PredictionUnavailable("pchip_knots_not_strictly_increasing")
    secants = [(ys[i + 1] - ys[i]) / h[i] for i in range(len(h))]
    if len(xs) == 2:
        return [secants[0], secants[0]]

    derivatives = [0.0] * len(xs)
    for index in range(1, len(xs) - 1):
        before = secants[index - 1]
        after = secants[index]
        if before == 0 or after == 0 or (before < 0) != (after < 0):
            derivatives[index] = 0.0
        else:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            derivatives[index] = (w1 + w2) / (w1 / before + w2 / after)

    def endpoint(h0: float, h1: float, delta0: float, delta1: float) -> float:
        value = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if value == 0 or delta0 == 0 or (value < 0) != (delta0 < 0):
            return 0.0
        if (delta0 < 0) != (delta1 < 0) and abs(value) > abs(3.0 * delta0):
            return 3.0 * delta0
        return value

    derivatives[0] = endpoint(h[0], h[1], secants[0], secants[1])
    derivatives[-1] = endpoint(h[-1], h[-2], secants[-1], secants[-2])
    return derivatives


def pchip_1d(xs: Iterable[float], ys: Iterable[float], x: float) -> float:
    """Evaluate shape-preserving cubic Hermite interpolation without extrapolation."""
    x_values = [float(value) for value in xs]
    y_values = [_finite(value, "pchip_y") for value in ys]
    x = _finite(x, "pchip_x")
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise PredictionUnavailable("pchip_needs_two_matching_knots")
    if any(right <= left for left, right in zip(x_values, x_values[1:])):
        raise PredictionUnavailable("pchip_knots_not_strictly_increasing")
    if x < x_values[0] or x > x_values[-1]:
        raise PredictionUnavailable(f"out_of_grid:{x}:{x_values[0]}:{x_values[-1]}")
    if any(value < 0 for value in y_values):
        raise PredictionUnavailable("negative_pchip_coefficient")
    derivatives = _pchip_derivatives(x_values, y_values)
    for index, knot in enumerate(x_values):
        if x == knot:
            return y_values[index]
    interval = next(i for i in range(len(x_values) - 1) if x_values[i] < x < x_values[i + 1])
    width = x_values[interval + 1] - x_values[interval]
    t = (x - x_values[interval]) / width
    t2 = t * t
    t3 = t2 * t
    y = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y_values[interval]
        + (t3 - 2.0 * t2 + t) * width * derivatives[interval]
        + (-2.0 * t3 + 3.0 * t2) * y_values[interval + 1]
        + (t3 - t2) * width * derivatives[interval + 1]
    )
    if not math.isfinite(y) or y < 0:
        raise PredictionUnavailable("invalid_interpolated_coefficient")
    return y


def _prefix_nonnegative_nodes(
    nodes: Mapping[int, float], label: str
) -> tuple[list[int], list[float]]:
    """Keep only a valid low-width prefix; never bridge an invalid knot."""
    ordered = sorted(nodes)
    valid_x: list[int] = []
    valid_y: list[float] = []
    saw_invalid = False
    for x in ordered:
        value = _finite(nodes[x], label)
        if value < 0:
            saw_invalid = True
            continue
        if saw_invalid:
            raise PredictionUnavailable(f"non_terminal_invalid_knot:{label}:{x}")
        valid_x.append(x)
        valid_y.append(value)
    if len(valid_x) < 2:
        raise PredictionUnavailable(f"insufficient_valid_knots:{label}")
    return valid_x, valid_y


def interpolate_surface(
    nodes: Mapping[tuple[int, int | None], float],
    n: int,
    chi: int | None,
    *,
    allow_invalid_statevector_sampling_tail: bool = False,
    label: str = "coefficient",
) -> float:
    """Interpolate a statevector n curve or MPS tensor-product surface.

    For MPS, PCHIP is first evaluated in n at each configured chi knot, then in
    log2(chi), which defines the tensor-product order used by this adaptation.
    """
    if chi is None:
        one_d: dict[int, float] = {}
        for (width, bond), value in nodes.items():
            if bond is None:
                one_d[width] = value
        if not one_d:
            raise PredictionUnavailable(f"missing_surface:{label}")
        if allow_invalid_statevector_sampling_tail:
            xs, ys = _prefix_nonnegative_nodes(one_d, label)
        else:
            if any(_finite(value, label) < 0 for value in one_d.values()):
                raise PredictionUnavailable(f"negative_coefficient:{label}")
            xs, ys = sorted(one_d), [one_d[x] for x in sorted(one_d)]
        return pchip_1d(xs, ys, float(n))

    if chi <= 0:
        raise PredictionUnavailable("configured_chi_must_be_positive")
    bonds = sorted({bond for _, bond in nodes if bond is not None})
    if not bonds:
        raise PredictionUnavailable(f"missing_surface:{label}")
    if chi < bonds[0] or chi > bonds[-1]:
        raise PredictionUnavailable(f"out_of_grid_chi:{chi}:{bonds[0]}:{bonds[-1]}")
    widths = sorted({width for (width, bond) in nodes if bond is not None})
    if len(widths) < 2 or len(bonds) < 2:
        raise PredictionUnavailable(f"incomplete_tensor_grid:{label}")
    for width in widths:
        for bond in bonds:
            key = (width, bond)
            if key not in nodes:
                raise PredictionUnavailable(f"missing_tensor_knot:{label}:{width}:{bond}")
            if _finite(nodes[key], label) < 0:
                raise PredictionUnavailable(f"negative_tensor_knot:{label}:{width}:{bond}")
    along_n = [pchip_1d(widths, [nodes[(width, bond)] for width in widths], n)
               for bond in bonds]
    log_bonds = [math.log2(bond) for bond in bonds]
    return pchip_1d(log_bonds, along_n, math.log2(chi))


def _interpolate_grid_axis(
    nodes: Mapping[float, float],
    expected_grid: Iterable[float],
    query: float,
    label: str,
) -> float:
    """Interpolate only inside a contiguous run of valid expected knots.

    Missing expected coordinates split the surface.  This prevents ordinary
    PCHIP from silently connecting valid knots on either side of a failed
    calibration cell.
    """
    grid = [float(value) for value in expected_grid]
    if len(grid) < 2 or any(right <= left for left, right in zip(grid, grid[1:])):
        raise PredictionUnavailable(f"invalid_expected_grid:{label}")
    if query < grid[0] or query > grid[-1]:
        raise PredictionUnavailable(f"out_of_grid:{query}:{grid[0]}:{grid[-1]}")

    valid: list[tuple[int, float]] = []
    for index, coordinate in enumerate(grid):
        if coordinate in nodes:
            value = _finite(nodes[coordinate], label)
            if value < 0:
                raise PredictionUnavailable(f"negative_coefficient:{label}:{coordinate}")
            valid.append((index, value))

    for index, value in valid:
        if query == grid[index]:
            return value
    brackets = [i for i in range(len(grid) - 1) if grid[i] < query < grid[i + 1]]
    if not brackets:
        raise PredictionUnavailable(f"missing_expected_knot:{label}:{query}")
    left_index = brackets[0]
    right_index = left_index + 1
    if grid[left_index] not in nodes or grid[right_index] not in nodes:
        raise PredictionUnavailable(
            f"invalid_or_missing_bracketing_knot:{label}:{grid[left_index]}:{grid[right_index]}"
        )

    # Find the contiguous valid segment containing the bracket so PCHIP's
    # endpoint derivatives cannot borrow information across an invalid knot.
    segment_start = left_index
    while segment_start > 0 and grid[segment_start - 1] in nodes:
        segment_start -= 1
    segment_end = right_index
    while segment_end + 1 < len(grid) and grid[segment_end + 1] in nodes:
        segment_end += 1
    xs = grid[segment_start:segment_end + 1]
    ys = [_finite(nodes[x], label) for x in xs]
    return pchip_1d(xs, ys, query)


def interpolate_surface_v2(
    nodes: Mapping[tuple[int, int | None], float],
    invalid_nodes: Mapping[tuple[int, int | None], str],
    n: int,
    chi: int | None,
    expected_widths: Iterable[int],
    expected_chis: Iterable[int] = (),
    *,
    label: str = "coefficient",
) -> float:
    """V2 PCHIP with explicit failed-knot checks and no gap bridging."""
    widths = tuple(int(value) for value in expected_widths)
    if chi is None:
        if (n, None) in invalid_nodes:
            raise PredictionUnavailable(f"unavailable_knot:{label}:{invalid_nodes[(n, None)]}")
        one_d = {width: value for (width, bond), value in nodes.items() if bond is None}
        if n in one_d:
            value = _finite(one_d[n], label)
            if value < 0:
                raise PredictionUnavailable(f"negative_coefficient:{label}:{n}")
            return value
        return _interpolate_grid_axis(one_d, widths, float(n), label)

    bonds = tuple(int(value) for value in expected_chis)
    if chi <= 0 or not bonds:
        raise PredictionUnavailable("configured_chi_must_be_positive")
    if chi < bonds[0] or chi > bonds[-1]:
        raise PredictionUnavailable(f"out_of_grid_chi:{chi}:{bonds[0]}:{bonds[-1]}")
    values_by_bond: dict[float, float] = {}
    invalid_by_bond: dict[float, str] = {}
    for bond in bonds:
        expected_key = (n, bond)
        if expected_key in invalid_nodes:
            invalid_by_bond[float(bond)] = invalid_nodes[expected_key]
            continue
        width_nodes = {width: value for (width, node_bond), value in nodes.items()
                       if node_bond == bond}
        try:
            values_by_bond[float(bond)] = _interpolate_grid_axis(
                width_nodes, widths, float(n), f"{label}:chi={bond}"
            )
        except PredictionUnavailable as exc:
            invalid_by_bond[float(bond)] = str(exc)
    if float(chi) in invalid_by_bond:
        raise PredictionUnavailable(
            f"unavailable_knot:{label}:chi={chi}:{invalid_by_bond[float(chi)]}"
        )
    if float(chi) in values_by_bond:
        return values_by_bond[float(chi)]
    log_nodes = {math.log2(bond): value for bond, value in values_by_bond.items()}
    log_grid = [math.log2(bond) for bond in bonds]
    return _interpolate_grid_axis(log_nodes, log_grid, math.log2(float(chi)), label)


def _component_nodes(
    calibration: FittedCalibration, candidate: str, component: str
) -> dict[tuple[int, int | None], float]:
    return {
        (width, chi): value
        for (candidate_id, component_id, width, chi), value in calibration.operation_knots.items()
        if candidate_id == candidate and component_id == component
    }


def _sampling_nodes(
    calibration: FittedCalibration, candidate: str, field: str
) -> dict[tuple[int, int | None], float]:
    source = (calibration.sampling_intercept_knots if field == "intercept"
              else calibration.sampling_per_shot_knots)
    return {(width, chi): value for (candidate_id, width, chi), value in source.items()
            if candidate_id == candidate}


@dataclass(frozen=True)
class CircuitOperationFeatures:
    n_qubits: int
    n_one_qubit_operations: int
    n_cx_operations: int
    measured_qubits: int
    unsupported_operations: Mapping[str, int]
    gate_counts: Mapping[str, int]

    @property
    def compatible(self) -> bool:
        return self.n_qubits > 0 and not self.unsupported_operations and self.measured_qubits == self.n_qubits


def parse_qasm_operation_features(qasm: str, expected_n_qubits: int | None = None) -> CircuitOperationFeatures:
    """Count QASM 2 native instruction arities used by the calibrated classes.

    Every one-qubit gate is charged to the calibrated 1Q class; only CX is
    accepted for the calibrated 2Q differential. Other 2Q/3Q gates, resets,
    conditionals and unsupported statements remain unavailable instead of
    being silently converted to CX equivalents. Measurements are represented
    by the candidate-local zero-gate sampling surface.
    """
    qregs: dict[str, int] = {}
    one_q = cx = 0
    measured_indices: set[tuple[str, int]] = set()
    unsupported: dict[str, int] = defaultdict(int)
    gate_counts: dict[str, int] = defaultdict(int)

    for raw_line in qasm.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("qreg "):
            match = re.match(r"qreg\s+([a-zA-Z_]\w*)\[(\d+)\]\s*;", line, re.I)
            if not match:
                unsupported["qreg_parse"] += 1
            else:
                qregs[match.group(1)] = int(match.group(2))
            continue
        if low.startswith(("openqasm", "include ", "creg ", "gate ", "opaque ")):
            continue
        if low.startswith("barrier"):
            continue
        if low.startswith("measure"):
            match = re.match(r"measure\s+(.+?)\s*->\s*(.+?)\s*;\s*$", line, re.I)
            if not match:
                unsupported["measure_parse"] += 1
                continue
            left = match.group(1).strip()
            qmatch = re.fullmatch(r"([a-zA-Z_]\w*)\[(\d+)\]", left)
            if qmatch:
                measured_indices.add((qmatch.group(1), int(qmatch.group(2))))
            elif left in qregs:
                measured_indices.update((left, index) for index in range(qregs[left]))
            else:
                unsupported["measure_register_unresolved"] += 1
            continue

        if low.startswith("if "):
            unsupported["classical_conditional"] += 1
            continue
        match = re.match(r"([a-zA-Z_]\w*)(?:\s*\([^;]*\))?\s+(.+?)\s*;\s*$", line)
        if not match:
            unsupported["qasm_statement_parse"] += 1
            continue
        operation = match.group(1).lower()
        refs = re.findall(r"([a-zA-Z_]\w*)\[(\d+)\]", match.group(2))
        arity = len(refs)
        gate_counts[operation] += 1
        if operation == "reset":
            unsupported["reset"] += 1
        elif arity == 1:
            one_q += 1
        elif arity == 2 and operation == "cx":
            cx += 1
        elif arity == 2:
            unsupported[f"two_qubit_{operation}"] += 1
        elif arity >= 3:
            unsupported[f"{arity}_qubit_{operation}"] += 1
        else:
            unsupported[f"unclassified_{operation}"] += 1

    n_qubits = sum(qregs.values())
    if expected_n_qubits is not None and n_qubits != int(expected_n_qubits):
        unsupported["panel_qasm_width_mismatch"] += 1
    measured_count = len(measured_indices)
    return CircuitOperationFeatures(
        n_qubits=n_qubits,
        n_one_qubit_operations=one_q,
        n_cx_operations=cx,
        measured_qubits=measured_count,
        unsupported_operations=dict(sorted(unsupported.items())),
        gate_counts=dict(sorted(gate_counts.items())),
    )


@dataclass(frozen=True)
class CandidateRuntimePrediction:
    candidate: str
    configured_chi: int | None
    shots: int
    n_qubits: int
    one_qubit_seconds: float
    cx_seconds: float
    sampling_seconds: float
    predicted_warm_seconds: float
    calibration_clock_id: str


@dataclass(frozen=True)
class CandidateRuntimePredictionV2:
    candidate: str
    configured_chi: int | None
    shots: int
    n_qubits: int
    one_qubit_seconds: float
    cx_seconds: float
    sampling_seconds: float
    predicted_process_isolated_seconds: float
    calibration_clock_id: str


@dataclass(frozen=True)
class CandidateRuntimePredictionV3:
    candidate: str
    configured_chi: int | None
    shots: int
    n_qubits: int
    one_qubit_seconds: float
    cx_seconds: float
    sampling_seconds: float
    predicted_process_isolated_seconds: float
    calibration_clock_id: str


def predict_candidate_runtime(
    calibration: FittedCalibration,
    features: CircuitOperationFeatures,
    candidate: str,
    shots: int,
    configured_chi: int | None = None,
    *,
    allow_invalid_sampling_tail: bool = True,
) -> CandidateRuntimePrediction:
    """Predict one candidate warm runtime without selector/oracle decisions."""
    if candidate not in {"statevector", "mps_fixed_chi"}:
        raise PredictionUnavailable(f"unknown_candidate:{candidate}")
    if candidate == "statevector" and configured_chi is not None:
        raise PredictionUnavailable("statevector_has_no_configured_chi")
    if candidate == "mps_fixed_chi" and configured_chi is None:
        raise PredictionUnavailable("mps_requires_preconfigured_chi")
    if shots <= 0:
        raise PredictionUnavailable("shots_must_be_positive")
    if not features.compatible:
        raise PredictionUnavailable(
            "incompatible_candidate_ir_features:"
            + ",".join(f"{key}={value}" for key, value in features.unsupported_operations.items())
            + (";incomplete_measurements" if features.measured_qubits != features.n_qubits else "")
        )
    model_candidate = "statevector" if candidate == "statevector" else "mps_fixed_chi"
    chi = configured_chi
    one_q = interpolate_surface(
        _component_nodes(calibration, model_candidate, "one_qubit"),
        features.n_qubits, chi, label="one_qubit",
    )
    cx = interpolate_surface(
        _component_nodes(calibration, model_candidate, "two_qubit_cx"),
        features.n_qubits, chi, label="two_qubit_cx",
    )
    intercept_nodes = _sampling_nodes(calibration, model_candidate, "intercept")
    per_shot_nodes = _sampling_nodes(calibration, model_candidate, "per_shot")
    intercept = interpolate_surface(intercept_nodes, features.n_qubits, chi, label="sampling_intercept")
    per_shot = interpolate_surface(
        per_shot_nodes, features.n_qubits, chi,
        allow_invalid_statevector_sampling_tail=(
            allow_invalid_sampling_tail and model_candidate == "statevector"
        ),
        label="sampling_per_shot",
    )

    if model_candidate == "statevector":
        one_seconds = one_q * (2 ** features.n_qubits) * features.n_one_qubit_operations
        cx_seconds = cx * (2 ** features.n_qubits) * features.n_cx_operations
    else:
        assert chi is not None
        one_seconds = one_q * (chi ** 2) * features.n_one_qubit_operations
        cx_seconds = cx * features.n_qubits * (chi ** 3) * features.n_cx_operations
    sampling_seconds = intercept + per_shot * shots
    total = one_seconds + cx_seconds + sampling_seconds
    if any(value < 0 or not math.isfinite(value)
           for value in (one_seconds, cx_seconds, sampling_seconds, total)):
        raise PredictionUnavailable("negative_or_non_finite_predicted_runtime_component")
    return CandidateRuntimePrediction(
        candidate=candidate,
        configured_chi=chi,
        shots=shots,
        n_qubits=features.n_qubits,
        one_qubit_seconds=one_seconds,
        cx_seconds=cx_seconds,
        sampling_seconds=sampling_seconds,
        predicted_warm_seconds=total,
        calibration_clock_id=calibration.clock_id,
    )


def predict_candidate_runtime_v2(
    calibration: FittedCalibration,
    features: CircuitOperationFeatures,
    candidate: str,
    shots: int,
    configured_chi: int | None = None,
) -> CandidateRuntimePredictionV2:
    """Predict with v2 no-extrapolation/no-invalid-knot-bridging semantics."""
    if calibration.fit_version != "process_isolated_affine_v2":
        raise PredictionUnavailable("v2_prediction_requires_v2_fit")
    if candidate not in {"statevector", "mps_fixed_chi"}:
        raise PredictionUnavailable(f"unknown_candidate:{candidate}")
    if candidate == "statevector" and configured_chi is not None:
        raise PredictionUnavailable("statevector_has_no_configured_chi")
    if candidate == "mps_fixed_chi" and configured_chi is None:
        raise PredictionUnavailable("mps_requires_preconfigured_chi")
    if shots <= 0:
        raise PredictionUnavailable("shots_must_be_positive")
    if not features.compatible:
        raise PredictionUnavailable("incompatible_candidate_ir_features")
    model_candidate = candidate
    chi = configured_chi
    widths = calibration.interpolation_width_grid.get(model_candidate, ())
    if not widths:
        raise PredictionUnavailable(f"missing_expected_width_grid:{model_candidate}")
    bonds = calibration.interpolation_chi_grid if model_candidate == "mps_fixed_chi" else ()

    def knot_map(component: str) -> tuple[dict[tuple[int, int | None], float],
                                            dict[tuple[int, int | None], str]]:
        if component == "one_qubit":
            nodes = _component_nodes(calibration, model_candidate, component)
        elif component == "two_qubit_cx":
            nodes = _component_nodes(calibration, model_candidate, component)
        elif component == "sampling_intercept":
            nodes = _sampling_nodes(calibration, model_candidate, "intercept")
        elif component == "sampling_per_shot":
            nodes = _sampling_nodes(calibration, model_candidate, "per_shot")
        else:  # pragma: no cover - fixed call sites below
            raise AssertionError(component)
        invalid = {(width, bond): reason for (candidate_id, component_id, width, bond), reason
                   in calibration.invalid_knot_states.items()
                   if candidate_id == model_candidate and component_id == component}
        return nodes, invalid

    def value(component: str) -> float:
        nodes, invalid = knot_map(component)
        return interpolate_surface_v2(nodes, invalid, features.n_qubits, chi,
                                       widths, bonds, label=component)

    one_q = value("one_qubit") if features.n_one_qubit_operations else 0.0
    # Circuits with no CX have no dependency on the CX calibration surface.
    # This matters for executable 1Q-only members when the CX knot is invalid.
    cx = value("two_qubit_cx") if features.n_cx_operations else 0.0
    intercept = value("sampling_intercept")
    per_shot = value("sampling_per_shot")
    if intercept < 0 or per_shot < 0:
        raise PredictionUnavailable("negative_sampling_coefficient")
    if candidate == "statevector":
        one_seconds = one_q * (2 ** features.n_qubits) * features.n_one_qubit_operations
        cx_seconds = cx * (2 ** features.n_qubits) * features.n_cx_operations
    else:
        assert chi is not None
        one_seconds = one_q * (chi ** 2) * features.n_one_qubit_operations
        cx_seconds = cx * features.n_qubits * (chi ** 3) * features.n_cx_operations
    sampling_seconds = intercept + per_shot * shots
    total = one_seconds + cx_seconds + sampling_seconds
    if any(not math.isfinite(component_value) or component_value < 0
           for component_value in (one_seconds, cx_seconds, sampling_seconds, total)):
        raise PredictionUnavailable("negative_or_non_finite_predicted_runtime_component")
    return CandidateRuntimePredictionV2(
        candidate=candidate,
        configured_chi=chi,
        shots=shots,
        n_qubits=features.n_qubits,
        one_qubit_seconds=one_seconds,
        cx_seconds=cx_seconds,
        sampling_seconds=sampling_seconds,
        predicted_process_isolated_seconds=total,
        calibration_clock_id=calibration.clock_id,
    )


def predict_candidate_runtime_v3(
    calibration: FittedCalibration,
    features: CircuitOperationFeatures,
    candidate: str,
    shots: int,
    configured_chi: int | None = None,
    *,
    _required_manifest_id: str = "maestro-common-panel-completion-v3",
    _required_fit_version: str = "paired_endpoint_heldout_512_v3",
) -> CandidateRuntimePredictionV3:
    """Predict one candidate runtime, without selector or oracle logic.

    The private protocol parameters let the S6 partial wrapper reuse the same
    numerical surface while requiring a distinct fitted-ledger identity.
    """
    if (calibration.calibration_manifest_id != _required_manifest_id
            or calibration.fit_version != _required_fit_version):
        raise PredictionUnavailable("calibration_protocol_or_fit_version_mismatch")
    if candidate not in {"statevector", "mps_fixed_chi"}:
        raise PredictionUnavailable(f"unknown_candidate:{candidate}")
    if candidate == "statevector" and configured_chi is not None:
        raise PredictionUnavailable("statevector_has_no_configured_chi")
    if candidate == "mps_fixed_chi" and configured_chi is None:
        raise PredictionUnavailable("mps_requires_preconfigured_chi")
    if shots <= 0:
        raise PredictionUnavailable("shots_must_be_positive")
    if not features.compatible:
        raise PredictionUnavailable("incompatible_candidate_ir_features")

    widths = calibration.interpolation_width_grid.get(candidate, ())
    if not widths:
        raise PredictionUnavailable(f"missing_expected_width_grid:{candidate}")
    bonds = calibration.interpolation_chi_grid if candidate == "mps_fixed_chi" else ()
    chi = configured_chi

    def operation_coefficient(component: str) -> float:
        nodes = _component_nodes(calibration, candidate, component)
        invalid = {
            (width, bond): reason
            for (candidate_id, component_id, width, bond), reason
            in calibration.invalid_knot_states.items()
            if candidate_id == candidate and component_id == component
        }
        return interpolate_surface_v2(
            nodes, invalid, features.n_qubits, chi, widths, bonds, label=component
        )

    def sampling_coefficient(field: str, component: str) -> float:
        nodes = _sampling_nodes(calibration, candidate, field)
        invalid = {
            (width, bond): reason
            for (candidate_id, component_id, width, bond), reason
            in calibration.invalid_knot_states.items()
            if candidate_id == candidate and component_id == component
        }
        return interpolate_surface_v2(
            nodes, invalid, features.n_qubits, chi, widths, bonds, label=component
        )

    # An unused component is mathematically absent and must not make an
    # otherwise executable circuit unavailable due to that surface's knots.
    one_q = operation_coefficient("one_qubit") if features.n_one_qubit_operations else 0.0
    cx = operation_coefficient("two_qubit_cx") if features.n_cx_operations else 0.0
    intercept = sampling_coefficient("intercept", "sampling_intercept")
    per_shot = sampling_coefficient("per_shot", "sampling_per_shot")
    if intercept < 0 or per_shot < 0:
        raise PredictionUnavailable("negative_sampling_coefficient")
    if candidate == "statevector":
        one_seconds = one_q * (2 ** features.n_qubits) * features.n_one_qubit_operations
        cx_seconds = cx * (2 ** features.n_qubits) * features.n_cx_operations
    else:
        assert chi is not None
        one_seconds = one_q * chi**2 * features.n_one_qubit_operations
        cx_seconds = cx * features.n_qubits * chi**3 * features.n_cx_operations
    sampling_seconds = intercept + per_shot * shots
    total = one_seconds + cx_seconds + sampling_seconds
    if any(not math.isfinite(value) or value < 0 for value in (
        one_seconds, cx_seconds, sampling_seconds, total
    )):
        raise PredictionUnavailable("negative_or_non_finite_predicted_runtime_component")
    return CandidateRuntimePredictionV3(
        candidate=candidate, configured_chi=chi, shots=shots,
        n_qubits=features.n_qubits, one_qubit_seconds=one_seconds,
        cx_seconds=cx_seconds, sampling_seconds=sampling_seconds,
        predicted_process_isolated_seconds=total,
        calibration_clock_id=calibration.clock_id,
    )


def predict_candidate_runtime_partial(
    calibration: FittedCalibration,
    features: CircuitOperationFeatures,
    candidate: str,
    shots: int,
    configured_chi: int | None = None,
) -> CandidateRuntimePredictionV3:
    """Predict only from the S6 partial-policy calibration ledger fit."""
    return predict_candidate_runtime_v3(
        calibration=calibration,
        features=features,
        candidate=candidate,
        shots=shots,
        configured_chi=configured_chi,
        _required_manifest_id=PARTIAL_CALIBRATION_MANIFEST_ID,
        _required_fit_version=PARTIAL_CALIBRATION_FIT_VERSION,
    )
