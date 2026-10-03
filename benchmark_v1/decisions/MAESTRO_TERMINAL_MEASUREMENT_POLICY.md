# Maestro terminal-measurement policy

Status: scientific scope frozen; no QCSim timings or model training performed.
This supersedes the prospective full-register-only eligibility rule in
`maestro_cpu_component_benchmark.json`, not any historical timing artifact.

## Dataset and admissible representation

Retain all 150 exact source hashes, their 162 aliases, and the existing five
folds. Admit nonempty terminal injective measurement maps, including partial
measurement and extra classical bits. Never add measurements to fill a register,
truncate classical registers, or move a measurement across a quantum operation.
Reject repeated measured qubits, repeated destination bits, and nonterminal
measurement. Terminal barriers are harmless. Check qubit/classical register
widths and the exact map before restoring source order after transpilation;
independent terminal measurements may be reordered by Qiskit. Verify the
unitary prefix on zero and two fixed-seed product inputs outside timing.
These finite probes are a numerical check, not a proof for arbitrary circuits.

Static source classification: 112 full terminal maps; 23 terminal partial maps;
8 QFT circuits with wider classical output; 7 repeated-measurement circuits. Expected
eligible count is 143/150 (95.33%), not an observed-runtime success count.
The seven portfolio-QAOA circuits measure the same qubits twice into distinct
classical registers, separated by a barrier, with no intervening quantum gate.
Earlier descriptions as nonterminal/mid-circuit measurement were inaccurate.
They remain outside this injective-map scope; their equivalence must not be
assumed by dropping a readout layer. Keep these hashes in the denominator and report actual timing
coverage separately. None of this changes the real-QPU dataset or V4 boundary.

## Sampling model: deliberately limited, still testable

Keep the 112 synthetic calibration cells and fixed component formula unchanged.
The paper's component/PCHIP/affine-shots structure motivates this adaptation:
https://arxiv.org/html/2512.04216v1#S3.SS3 . It does not establish that our local
zero-state coefficients generalize to every measurement layout or state.

Pinned QCSim source at revision `2ef7000395fc2a173ab1b4221ec9a235da0392c6`:
`Simulators/QCSimState.h:1821-1837` builds an alias table from the full state,
samples a basis index, and packs the requested measured bits. `Utils/Alias.h:
120-163` scans amplitudes and builds support-dependent tables. `Network/
NetworkJob.h:130-148` maps and aggregates distinct results; Python conversion
is outside the reported timer (`python/bindings.cpp:447-464`). Thus measured
count, output extent, state support and distinct-outcome count can affect cost.

The zero-state calibration does not identify those effects. This is a predictor
limitation, not a reason to exclude otherwise representable test circuits.
Do not invent an m/n correction, fit on test runtime, or select calibration
using accuracy. The primary remains explicitly an adaptation, not a faithful
reproduction or physically identified launch-overhead model. Report residuals
for the three static groups alongside the common-target comparison. Missing
features and low accuracy are acceptable benchmark findings.

Additional layout/high-support sampling controls would be a separately frozen
future intervention, not silently added to this primary calibration. They are
not required to begin the current, honestly scoped benchmark.

## Routine-agent execution handoff

1. Regenerate the static plan under ignored `work/` because protocol/runner
   hashes changed. Do not reuse a plan or pilot pinned to the superseded rule.
   Run the focused tests and static normalization audit: expect 143 admissible
   hashes and seven repeated-qubit measurement failures; stop on any other discrepancy.
2. Record measured count, declared classical width and exact indexed measurement
   map from the source in the plan/derived static inventory. Keep this outside
   timing. Do not assume declared width always equals QCSim result extent.
3. Run the frozen ten-cell pilot (300 calls, one timing worker). Check identity,
   finite runtimes and stability, not predictive accuracy. Preserve failed
   attempts. Only promote on the existing technical pilot gate.
4. Run full calibration and panel under the same context: 112 calibration cells
   plus all 150 panel identities, 15 attempts each. Unsupported circuits remain
   explicit terminal records. No concurrent heavy workload during timing.
5. Aggregate matching QCSim labels; train the three-seed/five-fold graph
   comparator on CUDA, with classical baselines using the same eligible labels
   and frozen folds. No Aer/CUDA-Q label substitution. Respect train-only fitting.
6. Report envelope, common successful intersection, and measurement-group
   residuals, failures and coverage. Publish only final validated artifacts;
   leave partial ledgers in ignored work. No new human signature required.

Parallelize static packaging/documentation only. Timing is exclusive; comparison
training starts after measurement labels are frozen. No training/timing is
executed as part of this strong-agent policy review.
