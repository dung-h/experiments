# Ma--Li physical-DAG static baseline evaluation

This is the first gate before training a graph model. All models predict
the logged observed Ma--Li `result.time_taken` target; only the input
features differ. Fold assignments are grouped by raw-QASM SHA-256 and
strict splits also hold out the backend.

| Split | Model | MAE (s) | MedAE (s) | RMSE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|---:|---:|
| grouped_qasm_fold_1 | `qcre` | 0.6831 | 0.4261 | 0.9568 | 0.5239 | 0.5075 |
| grouped_qasm_fold_1 | `compiled` | 0.5751 | 0.2637 | 0.8276 | 0.6281 | 0.6315 |
| grouped_qasm_fold_1 | `physical_graph` | 0.5445 | 0.4249 | 0.7481 | 0.6818 | 0.6989 |
| grouped_qasm_fold_1 | `rich_summary` | 0.4949 | 0.3463 | 0.7066 | 0.7033 | 0.7314 |
| grouped_qasm_fold_2 | `qcre` | 0.8927 | 0.7599 | 1.1698 | 0.7956 | 0.7584 |
| grouped_qasm_fold_2 | `compiled` | 0.6865 | 0.5365 | 0.8833 | 0.8867 | 0.8622 |
| grouped_qasm_fold_2 | `physical_graph` | 0.5204 | 0.3797 | 0.6844 | 0.9367 | 0.9173 |
| grouped_qasm_fold_2 | `rich_summary` | 0.4418 | 0.3286 | 0.5985 | 0.9516 | 0.9367 |
| grouped_qasm_fold_3 | `qcre` | 0.8420 | 0.6544 | 1.1497 | 0.6919 | 0.5845 |
| grouped_qasm_fold_3 | `compiled` | 0.7085 | 0.6208 | 0.9350 | 0.7709 | 0.7253 |
| grouped_qasm_fold_3 | `physical_graph` | 0.5615 | 0.4449 | 0.7093 | 0.8679 | 0.8419 |
| grouped_qasm_fold_3 | `rich_summary` | 0.4802 | 0.4163 | 0.6021 | 0.9018 | 0.8861 |
| grouped_qasm_fold_4 | `qcre` | 0.8153 | 0.6998 | 1.0170 | 0.7199 | 0.6959 |
| grouped_qasm_fold_4 | `compiled` | 0.6370 | 0.6126 | 0.7968 | 0.8489 | 0.8133 |
| grouped_qasm_fold_4 | `physical_graph` | 0.5210 | 0.4033 | 0.6495 | 0.9171 | 0.8760 |
| grouped_qasm_fold_4 | `rich_summary` | 0.3762 | 0.2654 | 0.5119 | 0.9498 | 0.9230 |
| grouped_qasm_fold_5 | `qcre` | 0.8984 | 0.7452 | 1.1547 | 0.7207 | 0.6369 |
| grouped_qasm_fold_5 | `compiled` | 0.8345 | 0.6794 | 1.1339 | 0.7730 | 0.6499 |
| grouped_qasm_fold_5 | `physical_graph` | 0.7396 | 0.4308 | 1.3664 | 0.6691 | 0.4916 |
| grouped_qasm_fold_5 | `rich_summary` | 0.6805 | 0.4609 | 1.1861 | 0.7646 | 0.6169 |
| strict_osaka_fold_1 | `qcre` | 0.7051 | 0.4657 | 0.9456 | 0.6391 | 0.5888 |
| strict_osaka_fold_1 | `compiled` | 0.5719 | 0.4022 | 0.7711 | 0.7663 | 0.7265 |
| strict_osaka_fold_1 | `physical_graph` | 0.5525 | 0.4174 | 0.6978 | 0.8162 | 0.7760 |
| strict_osaka_fold_1 | `rich_summary` | 0.4698 | 0.3954 | 0.6172 | 0.8550 | 0.8248 |
| strict_osaka_fold_2 | `qcre` | 0.8762 | 0.6687 | 1.1704 | 0.7793 | 0.7595 |
| strict_osaka_fold_2 | `compiled` | 0.6490 | 0.6019 | 0.8453 | 0.8873 | 0.8746 |
| strict_osaka_fold_2 | `physical_graph` | 0.5225 | 0.4413 | 0.6572 | 0.9323 | 0.9242 |
| strict_osaka_fold_2 | `rich_summary` | 0.4586 | 0.3447 | 0.5935 | 0.9310 | 0.9382 |
| strict_osaka_fold_3 | `qcre` | 0.8234 | 0.5802 | 1.1715 | 0.6740 | 0.5494 |
| strict_osaka_fold_3 | `compiled` | 0.6963 | 0.5413 | 0.9469 | 0.7616 | 0.7056 |
| strict_osaka_fold_3 | `physical_graph` | 0.5307 | 0.4391 | 0.6783 | 0.8741 | 0.8490 |
| strict_osaka_fold_3 | `rich_summary` | 0.4464 | 0.3197 | 0.5964 | 0.9008 | 0.8832 |
| strict_osaka_fold_4 | `qcre` | 0.8364 | 0.6979 | 1.0540 | 0.7064 | 0.6963 |
| strict_osaka_fold_4 | `compiled` | 0.6677 | 0.5875 | 0.8575 | 0.8222 | 0.7989 |
| strict_osaka_fold_4 | `physical_graph` | 0.5832 | 0.5303 | 0.7206 | 0.8992 | 0.8580 |
| strict_osaka_fold_4 | `rich_summary` | 0.4705 | 0.4062 | 0.6064 | 0.9303 | 0.8994 |
| strict_osaka_fold_5 | `qcre` | 0.8836 | 0.7930 | 1.1047 | 0.7033 | 0.6529 |
| strict_osaka_fold_5 | `compiled` | 0.7933 | 0.6840 | 1.0484 | 0.7694 | 0.6874 |
| strict_osaka_fold_5 | `physical_graph` | 0.7172 | 0.4086 | 1.2763 | 0.6965 | 0.5367 |
| strict_osaka_fold_5 | `rich_summary` | 0.7091 | 0.4350 | 1.2036 | 0.7346 | 0.5880 |
| strict_kyoto_fold_1 | `qcre` | 0.6616 | 0.4691 | 0.9676 | 0.1675 | 0.3183 |
| strict_kyoto_fold_1 | `compiled` | 0.6145 | 0.3039 | 0.9293 | 0.1552 | 0.3712 |
| strict_kyoto_fold_1 | `physical_graph` | 0.5507 | 0.3617 | 0.8320 | 0.2495 | 0.4959 |
| strict_kyoto_fold_1 | `rich_summary` | 0.5923 | 0.2905 | 0.8672 | 0.1679 | 0.4524 |
| strict_kyoto_fold_2 | `qcre` | 0.9269 | 0.7942 | 1.1680 | 0.8104 | 0.7549 |
| strict_kyoto_fold_2 | `compiled` | 0.7449 | 0.5769 | 0.9562 | 0.8758 | 0.8358 |
| strict_kyoto_fold_2 | `physical_graph` | 0.5367 | 0.3055 | 0.7416 | 0.9324 | 0.9012 |
| strict_kyoto_fold_2 | `rich_summary` | 0.5015 | 0.3985 | 0.6563 | 0.9473 | 0.9226 |
| strict_kyoto_fold_3 | `qcre` | 0.8546 | 0.7364 | 1.0901 | 0.7192 | 0.6413 |
| strict_kyoto_fold_3 | `compiled` | 0.7444 | 0.6862 | 0.9412 | 0.7812 | 0.7326 |
| strict_kyoto_fold_3 | `physical_graph` | 0.6028 | 0.5787 | 0.7484 | 0.8573 | 0.8309 |
| strict_kyoto_fold_3 | `rich_summary` | 0.5766 | 0.5530 | 0.6864 | 0.8755 | 0.8578 |
| strict_kyoto_fold_4 | `qcre` | 0.7493 | 0.5273 | 0.9523 | 0.7373 | 0.7034 |
| strict_kyoto_fold_4 | `compiled` | 0.5810 | 0.4648 | 0.7133 | 0.8772 | 0.8336 |
| strict_kyoto_fold_4 | `physical_graph` | 0.4460 | 0.3089 | 0.5655 | 0.9329 | 0.8954 |
| strict_kyoto_fold_4 | `rich_summary` | 0.3310 | 0.1920 | 0.4679 | 0.9522 | 0.9284 |
| strict_kyoto_fold_5 | `qcre` | 0.8957 | 0.7507 | 1.2025 | 0.7386 | 0.6255 |
| strict_kyoto_fold_5 | `compiled` | 0.8796 | 0.6352 | 1.2371 | 0.7617 | 0.6036 |
| strict_kyoto_fold_5 | `physical_graph` | 0.7799 | 0.4738 | 1.4532 | 0.6441 | 0.4531 |
| strict_kyoto_fold_5 | `rich_summary` | 0.6998 | 0.5011 | 1.2351 | 0.7629 | 0.6050 |
| family_qft | `qcre` | 1.0921 | 0.9901 | 1.3511 | -0.7872 | -0.7206 |
| family_qft | `compiled` | 0.9180 | 0.8373 | 1.1363 | -0.2195 | -0.2171 |
| family_qft | `physical_graph` | 0.6866 | 0.6653 | 0.8423 | 0.3401 | 0.3312 |
| family_qft | `rich_summary` | 0.5460 | 0.4944 | 0.6945 | 0.5502 | 0.5454 |
| family_qftentangled | `qcre` | 1.1395 | 1.0739 | 1.4105 | -0.9055 | -0.7732 |
| family_qftentangled | `compiled` | 0.8186 | 0.6975 | 1.0218 | 0.0878 | 0.0694 |
| family_qftentangled | `physical_graph` | 0.6926 | 0.5892 | 0.8902 | 0.3184 | 0.2936 |
| family_qftentangled | `rich_summary` | 0.5992 | 0.5314 | 0.7428 | 0.5075 | 0.5082 |
| family_qnn | `qcre` | 1.9700 | 1.9239 | 2.0151 | -5.8693 | -5.6687 |
| family_qnn | `compiled` | 0.8464 | 0.9372 | 0.9901 | -0.8385 | -0.6099 |
| family_qnn | `physical_graph` | 0.3918 | 0.2299 | 0.7037 | 0.2145 | 0.1867 |
| family_qnn | `rich_summary` | 0.3353 | 0.2160 | 0.6238 | 0.4288 | 0.3609 |
| family_qpeexact | `qcre` | 0.9083 | 0.6196 | 1.2685 | -0.8457 | -1.0415 |
| family_qpeexact | `compiled` | 0.7150 | 0.4929 | 0.9798 | -0.1668 | -0.2180 |
| family_qpeexact | `physical_graph` | 0.5904 | 0.4404 | 0.7319 | 0.3345 | 0.3204 |
| family_qpeexact | `rich_summary` | 0.5600 | 0.4817 | 0.7002 | 0.3984 | 0.3780 |
| family_qpeinexact | `qcre` | 0.8753 | 0.6291 | 1.2321 | -1.2171 | -1.3988 |
| family_qpeinexact | `compiled` | 0.6575 | 0.3864 | 0.9470 | -0.3824 | -0.4170 |
| family_qpeinexact | `physical_graph` | 0.5069 | 0.3941 | 0.6558 | 0.3028 | 0.3205 |
| family_qpeinexact | `rich_summary` | 0.4274 | 0.3170 | 0.5743 | 0.4738 | 0.4788 |
| family_qwalk_noancilla | `qcre` | 3.3807 | 3.3807 | 3.3945 | -363.3684 | -449.5806 |
| family_qwalk_noancilla | `compiled` | 4.0018 | 4.0018 | 4.0163 | -862.6540 | -629.7689 |
| family_qwalk_noancilla | `physical_graph` | 6.7738 | 6.7738 | 6.7777 | -3253.6691 | -1795.2747 |
| family_qwalk_noancilla | `rich_summary` | 5.5499 | 5.5499 | 5.5580 | -1916.2945 | -1206.9358 |
| family_random | `qcre` | 0.6793 | 0.5042 | 0.9082 | 0.6236 | 0.4974 |
| family_random | `compiled` | 0.7348 | 0.7544 | 0.8884 | 0.5610 | 0.5191 |
| family_random | `physical_graph` | 0.3066 | 0.1815 | 0.4030 | 0.9031 | 0.9010 |
| family_random | `rich_summary` | 1.1033 | 1.0582 | 1.1793 | 0.0795 | 0.1525 |
| family_realamprandom+twolocalrandom | `qcre` | 0.5125 | 0.3839 | 0.6562 | 0.6158 | 0.5979 |
| family_realamprandom+twolocalrandom | `compiled` | 0.4851 | 0.3475 | 0.6282 | 0.6297 | 0.6314 |
| family_realamprandom+twolocalrandom | `physical_graph` | 0.6085 | 0.5555 | 0.7632 | 0.5038 | 0.4561 |
| family_realamprandom+twolocalrandom | `rich_summary` | 0.4585 | 0.3439 | 0.5939 | 0.6907 | 0.6706 |
| family_su2random | `qcre` | 0.6547 | 0.6595 | 0.7829 | 0.4675 | 0.4655 |
| family_su2random | `compiled` | 0.6711 | 0.6143 | 0.7845 | 0.4395 | 0.4632 |
| family_su2random | `physical_graph` | 0.6305 | 0.5970 | 0.7246 | 0.5302 | 0.5421 |
| family_su2random | `rich_summary` | 0.4662 | 0.3469 | 0.6089 | 0.6572 | 0.6767 |

## Pooled interpretation

| Evaluation | Model | MAE (s) | log-R² | seconds-R² |
|---|---|---:|---:|---:|
| grouped_qasm | `qcre` | 0.8274 | 0.7321 | 0.6707 |
| grouped_qasm | `compiled` | 0.6889 | 0.8202 | 0.7660 |
| grouped_qasm | `physical_graph` | 0.5766 | 0.8527 | 0.7925 |
| grouped_qasm | `rich_summary` | 0.4937 | 0.8896 | 0.8433 |
| strict_all | `qcre` | 0.8229 | 0.7314 | 0.6744 |
| strict_all | `compiled` | 0.6924 | 0.8156 | 0.7625 |
| strict_all | `physical_graph` | 0.5817 | 0.8500 | 0.7916 |
| strict_all | `rich_summary` | 0.5230 | 0.8755 | 0.8306 |
| strict_osaka | `qcre` | 0.8258 | 0.7268 | 0.6776 |
| strict_osaka | `compiled` | 0.6755 | 0.8252 | 0.7828 |
| strict_osaka | `physical_graph` | 0.5795 | 0.8668 | 0.8127 |
| strict_osaka | `rich_summary` | 0.5088 | 0.8874 | 0.8459 |
| strict_kyoto | `qcre` | 0.8192 | 0.7357 | 0.6670 |
| strict_kyoto | `compiled` | 0.7144 | 0.8003 | 0.7325 |
| strict_kyoto | `physical_graph` | 0.5844 | 0.8246 | 0.7607 |
| strict_kyoto | `rich_summary` | 0.5414 | 0.8573 | 0.8081 |
| family_all | `qcre` | 0.9157 | 0.6207 | 0.5965 |
| family_all | `compiled` | 0.7228 | 0.8021 | 0.7490 |
| family_all | `physical_graph` | 0.6136 | 0.8351 | 0.7736 |
| family_all | `rich_summary` | 0.5752 | 0.8700 | 0.8117 |

The grouped-QASM pooled result is **not** a universal-device score.
The strict rows are two correlated held-backend diagnostics, with
Osaka and Kyoto reported separately in `summary.json`.
Family rows are connected-component leave-one-family-out diagnostics;
the component-level values are retained in `metrics.csv` because
QWalk has only two observations.

## QWalk guard

QWalk is only two rows and is not used as a standalone R². In the
grouped fold containing QWalk, the compact physical-count model has
absolute errors of about 6.95 s (Osaka) and 6.51 s (Kyoto); the richer
descriptor model lowers them to about 5.91 s and 5.35 s. This is an
improvement but still does not pass the protocol's tail/QWalk guard
by itself. A future DAG residual must address this explicitly.

The `qcre` row is a one-feature gate-aware weighted critical-path
calibration. `compiled` adds depth and active physical width. The
`physical_graph` row adds compact node/edge counts and `rich_summary`
adds timing/layer/opcode quantiles. These are static baselines, not
measured historical physical circuits. A DAG model is
only warranted if it improves these rows on the same splits and does
not worsen QWalk/tail error.
