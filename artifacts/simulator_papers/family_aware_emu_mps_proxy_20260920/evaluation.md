# Family-aware EMU-MPS threshold proxy

## Scope

- Conceptual reference: family-aware residual threshold prediction in
  arXiv:2606.11620.
- Target: smallest allowed EMU-MPS `max_bond_dim` rung attaining fidelity 0.99
  against a local high-χ reference; it is **not** wall-clock runtime.
- Corpus: 12 labelled local pulse sequences over four known pulse families,
  sizes 8, 12 and 16.
- Split: leave one size out. The held-out split has every family in train;
  this is known-family size transfer only.

| Method | N | Exact rung | Within ±1 rung | Mean abs. rung error |
|---|---:|---:|---:|---:|
| Shared numeric | 12 | 41.7% | 100.0% | 0.583 |
| Inferred-family residual | 12 | 41.7% | 100.0% | 0.583 |

The classifier inferred the family with **100.0%** accuracy in the same nested folds. Since the families are deliberately constructed with visibly distinct pulse metadata, that value is a data-property check, not a machine-learning claim.

## Boundary

The result cannot be compared numerically with the paper's QASM runtime and
approximation-threshold results. It merely checks whether the proposed
family-classification → residual-correction pattern adds value to the existing
local EMU-MPS threshold pilot without using the true test family label.
