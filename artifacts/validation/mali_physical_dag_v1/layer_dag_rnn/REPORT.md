# Ma--Li coarsened layer-DAG residual pilot

This pilot keeps ordered compiled-DAG timing structure while avoiding a
67-million-node PyG batch. Native nodes are aggregated into at most
**256** ordered bins and a small GRU predicts the residual
around the QCRE weighted critical path. It is not a full native-operation
GNN and uses the current FakeBackend reconstruction.

Device: **cuda**; seed: **1234**; early stopping uses a
group-disjoint inner validation split inside each outer training set.

| Split | MAE (s) | MedAE (s) | log-R² | seconds-R² |
|---|---:|---:|---:|---:|
| grouped_qasm_fold_1 | 0.8090 | 0.7504 | 0.3498 | 0.4053 |
| grouped_qasm_fold_2 | 0.8553 | 0.7238 | 0.8032 | 0.8013 |
| grouped_qasm_fold_3 | 0.7842 | 0.7502 | 0.7740 | 0.7143 |
| grouped_qasm_fold_4 | 0.8735 | 0.8186 | 0.7635 | 0.6859 |
| grouped_qasm_fold_5 | 0.9713 | 0.7543 | 0.6663 | 0.4865 |
| strict_osaka_fold_1 | 0.8371 | 0.6625 | 0.5514 | 0.4838 |
| strict_osaka_fold_2 | 0.9441 | 0.8754 | 0.7972 | 0.7723 |
| strict_osaka_fold_3 | 0.7857 | 0.6947 | 0.7754 | 0.7217 |
| strict_osaka_fold_4 | 0.9414 | 0.9157 | 0.7463 | 0.6559 |
| strict_osaka_fold_5 | 1.1506 | 0.9767 | 0.3455 | 0.3195 |
| strict_kyoto_fold_1 | 0.8562 | 0.7322 | -0.6505 | -0.0120 |
| strict_kyoto_fold_2 | 0.8173 | 0.5834 | 0.8617 | 0.7922 |
| strict_kyoto_fold_3 | 0.9108 | 0.7760 | 0.7079 | 0.6160 |
| strict_kyoto_fold_4 | 0.7761 | 0.7045 | 0.8015 | 0.6960 |
| strict_kyoto_fold_5 | 0.9627 | 0.7303 | 0.6973 | 0.5176 |

This is a pilot architecture result. It must be compared with the
static physical-summary baseline in the same artifact, and it must
pass the QWalk/tail guard and multiple-seed stability check before a
claim about DAG topology is made.
