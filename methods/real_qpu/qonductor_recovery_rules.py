"""Label-free candidate-width rules, not a logical-instance recovery engine.

The bulk inventory must supply AST-resolved terminal measurement pairs. This
module deliberately does not parse QASM, use backend capacity, or accept labels.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


CONTRACT = Path(__file__).resolve().parents[1] / "protocol/qonductor_logical_recovery.json"
WIDTH_RULES = json.loads(CONTRACT.read_text())["width_rules"]


def candidate_widths(
    family: str,
    registers: dict[str, int],
    measurements: list[tuple[int, str, int]],
    *,
    terminal_only: bool,
    has_control_flow: bool = False,
    has_reset: bool = False,
    archive_width: int | None = None,
) -> dict:
    """Return a width hypothesis from the pinned generator's measurement shape.

    Pairs are (physical quantum-wire index, classical register name, bit index).
    Absence of a hypothesis is investigation-required, not impossible recovery.
    Even a unique width is unverified until generator and mapping checks pass.
    """
    result = {"candidate_widths": [], "instance_status": "candidate_unverified",
              "reason": "investigation_required", "archive_width": archive_width}
    if has_control_flow or has_reset or not terminal_only:
        result["reason"] = "unsupported_measurement_lifecycle"
        return result
    if not measurements:
        result["reason"] = "no_measurement_pairs"
        return result
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
           for size in registers.values()):
        result["reason"] = "invalid_register_allocation"
        return result
    for wire, register, bit in measurements:
        if (not isinstance(wire, int) or wire < 0 or register not in registers
                or not isinstance(bit, int) or not 0 <= bit < registers[register]):
            result["reason"] = "invalid_measurement_operand"
            return result
    wires = [pair[0] for pair in measurements]
    destinations = [(pair[1], pair[2]) for pair in measurements]
    if len(set(wires)) != len(wires) or len(set(destinations)) != len(destinations):
        result["reason"] = "non_bijective_measurements"
        return result
    written = Counter(pair[1] for pair in measurements)
    if len(written) != 1:
        result["reason"] = "unreviewed_multiple_written_registers"
        return result
    register, count = next(iter(written.items()))
    if count != registers[register]:
        result["reason"] = "partially_written_register"
        return result
    rules = WIDTH_RULES
    if family == "qft":
        if len(registers) != 2 or set(registers.values()) != {count}:
            result["reason"] = "qft_generator_register_shape_mismatch"
            return result
        width = count
        reason = "qft_written_register_not_total_classical_allocation"
    elif family in rules["measured_all"] or family in rules["measured_plus_one"]:
        if len(registers) != 1:
            result["reason"] = "generator_register_shape_mismatch"
            return result
        offset = int(family in rules["measured_plus_one"])
        width = count + offset
        reason = "pinned_generator_measurement_count_plus_" + str(offset)
    else:
        result["reason"] = "family_rule_not_reviewed"
        return result
    if archive_width is not None and archive_width != width:
        result["reason"] = "archive_measurement_width_conflict"
        return result
    result.update(candidate_widths=[width], reason=reason)
    return result


def repeatability_pass(record: dict, repetitions: int) -> bool:
    """Unlike the old classifier, require every requested repetition to succeed."""
    runs = record.get("runs", [])
    return (repetitions > 0 and len(runs) == repetitions
            and record.get("all_runs_ok") is True
            and all(run.get("status") == "ok" for run in runs)
            and record.get("parameterized_reproducible") is True
            and record.get("structurally_reproducible") is True)
