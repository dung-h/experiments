"""Pinned reconstructions of Pasqal's documented EMU-MPS benchmark programs.

This module intentionally contains no locally invented pulse family.  The two
constructors transcribe the `Sequences used` snippets in the upstream pinned
``docs/emu_mps/benchmarks/index.md``.  Callers must record both this file hash
and the upstream document hash from the frozen input manifest.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.devices import AnalogDevice, MockDevice
from pulser.waveforms import RampWaveform


Family = Literal["adiabatic_afm", "quench"]


def adiabatic_afm(rows: int, columns: int) -> Sequence:
    """Official EMU-MPS benchmark's adiabatic 2D AFM sequence."""
    omega_max = 2.0 * 2 * math.pi
    interaction = omega_max / 2.0
    delta_0, delta_f = -6 * interaction, 2 * interaction
    t_rise, t_fall = 500, 1000
    t_sweep = int((delta_f - delta_0) / (2 * math.pi * 10) * 3000)
    spacing = MockDevice.rydberg_blockade_radius(interaction)
    register = Register.rectangle(rows, columns, spacing, prefix="q")
    sequence = Sequence(register, MockDevice)
    sequence.declare_channel("ising", "rydberg_global")
    sequence.add(Pulse.ConstantDetuning(RampWaveform(t_rise, 0.0, omega_max), delta_0, 0.0), "ising")
    sequence.add(Pulse.ConstantAmplitude(omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0), "ising")
    sequence.add(Pulse.ConstantDetuning(RampWaveform(t_fall, omega_max, 0.0), delta_f, 0.0), "ising")
    return sequence


def quench(rows: int, columns: int) -> Sequence:
    """Official EMU-MPS benchmark's 2D quench sequence."""
    hx, hz, evolution_time = 1.5, 0.0, 1.5
    spacing = 7.0
    register = Register.rectangle(rows, columns, spacing, prefix="q")
    interaction = AnalogDevice.interaction_coeff / spacing**6
    nearest_neighbor = interaction / 4
    omega = 2 * hx * nearest_neighbor
    delta = -2 * hz * nearest_neighbor + 2 * interaction
    duration = int(np.round(1000 * evolution_time / nearest_neighbor))
    sequence = Sequence(register, MockDevice)
    sequence.declare_channel("ising", "rydberg_global")
    sequence.add(Pulse.ConstantPulse(duration, omega, delta, 0), "ising")
    return sequence


def source_sequence(family: Family, rows: int, columns: int) -> Sequence:
    if family == "adiabatic_afm":
        return adiabatic_afm(rows, columns)
    if family == "quench":
        return quench(rows, columns)
    raise ValueError(f"unknown official source family: {family}")
