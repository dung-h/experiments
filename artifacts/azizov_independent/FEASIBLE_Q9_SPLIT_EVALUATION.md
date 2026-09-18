# Feasible subset split evaluation

This report uses the empirically feasible 149-circuit subset (1,192
successful runtime rows). The target is `log1p(T_exec_s)`. All models
use the same split within a section; the grouped circuit split is the
primary estimate because each logical circuit has multiple backend/opt
rows.

| Split | Block | Best model by log-R² | R² log | R² seconds | RMSE seconds |
|---|---|---|---:|---:|---:|
| random_row | source | svr | 0.7331 | 0.4763 | 0.6529 |
| random_row | compiled | random_forest | 0.7456 | 0.5438 | 0.6093 |
| random_row | hybrid | random_forest | 0.7635 | 0.5430 | 0.6098 |
| grouped_circuit | source | svr | 0.8200 | 0.8667 | 0.2173 |
| grouped_circuit | compiled | random_forest | 0.8135 | 0.8006 | 0.2657 |
| grouped_circuit | hybrid | random_forest | 0.8558 | 0.8230 | 0.2504 |
| family_held_out | source | svr | 0.7884 | 0.6326 | 0.4683 |
| family_held_out | compiled | random_forest | 0.7973 | 0.7700 | 0.3705 |
| family_held_out | hybrid | random_forest | 0.8375 | 0.7923 | 0.3521 |
| backend_held_out_FakeSherbrooke | source | random_forest | 0.5845 | 0.5308 | 0.5491 |
| backend_held_out_FakeSherbrooke | compiled | random_forest | 0.5308 | 0.5295 | 0.5498 |
| backend_held_out_FakeSherbrooke | hybrid | random_forest | 0.5511 | 0.5337 | 0.5473 |
| backend_held_out_FakeWashingtonV2 | source | svr | 0.5511 | 0.3357 | 0.4511 |
| backend_held_out_FakeWashingtonV2 | compiled | svr | 0.5255 | 0.4965 | 0.3927 |
| backend_held_out_FakeWashingtonV2 | hybrid | svr | 0.6297 | 0.5501 | 0.3712 |

## Interpretation

- Random-row scores are optimistic because the same logical circuit can
  appear in train and test under another backend or optimization level.
- Grouped-circuit scores are the primary within-domain estimate.
- Family-held-out scores test algorithm-family transfer, while the two
  backend-held-out scores test transfer between current fake snapshots.
- These are baseline results on a conservative local subset, not the
  paper's full 1,402-circuit/GNN result.
