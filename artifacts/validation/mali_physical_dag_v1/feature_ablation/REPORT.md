# Ma--Li compact feature ablation

This table attributes the improvement over the one-feature QCRE proxy.
Every block uses the same outer folds, train-only preprocessing and
observed `result.time_taken` target. It is an ablation, not evidence that
the current FakeBackend timing is historical hardware truth.

| Evaluation | Block | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|
| grouped-QASM | `qcre` | 0.8274 | 0.7321 | 0.6707 |
| grouped-QASM | `compiled` | 0.6889 | 0.8202 | 0.7660 |
| grouped-QASM | `graph_size` | 0.5766 | 0.8527 | 0.7925 |
| grouped-QASM | `layer` | 0.5635 | 0.8616 | 0.8022 |
| grouped-QASM | `timing` | 0.5109 | 0.8892 | 0.8446 |
| grouped-QASM | `edge` | 0.5766 | 0.8527 | 0.7925 |
| grouped-QASM | `opcode` | 0.5563 | 0.8241 | 0.7656 |
| grouped-QASM | `rich_summary` | 0.4937 | 0.8895 | 0.8432 |
| strict backend+QASM | `qcre` | 0.8229 | 0.7314 | 0.6744 |
| strict backend+QASM | `compiled` | 0.6924 | 0.8156 | 0.7625 |
| strict backend+QASM | `graph_size` | 0.5817 | 0.8500 | 0.7916 |
| strict backend+QASM | `layer` | 0.5760 | 0.8558 | 0.7972 |
| strict backend+QASM | `timing` | 0.5355 | 0.8732 | 0.8329 |
| strict backend+QASM | `edge` | 0.5817 | 0.8500 | 0.7916 |
| strict backend+QASM | `opcode` | 0.5746 | 0.8230 | 0.7666 |
| strict backend+QASM | `rich_summary` | 0.5230 | 0.8755 | 0.8306 |
| family held-out | `qcre` | 0.9157 | 0.6207 | 0.5965 |
| family held-out | `compiled` | 0.7228 | 0.8021 | 0.7490 |
| family held-out | `graph_size` | 0.6136 | 0.8351 | 0.7736 |
| family held-out | `layer` | 0.6035 | 0.8440 | 0.7849 |
| family held-out | `timing` | 0.5810 | 0.8615 | 0.8262 |
| family held-out | `edge` | 0.6136 | 0.8351 | 0.7736 |
| family held-out | `opcode` | 0.6678 | 0.7896 | 0.7096 |
| family held-out | `rich_summary` | 0.5753 | 0.8700 | 0.8117 |

## Reading the result

`compiled` isolates depth and active width; `graph_size` adds native node/edge counts;
`layer`, `timing`, `edge` and `opcode` add one interpretable block at a time.
`rich_summary` is the all-descriptor reference. A gain is meaningful only if it
survives grouped and strict splits; family-held-out values are a stress test.
QWalk is inspected separately because it has only two rows.
