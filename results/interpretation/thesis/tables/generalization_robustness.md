> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../../docs/thesis/README.md).

# Bảng 3. Generalization theo từng case

42 case×method rows, 21 cases; không pool 6684 assignments thành IID dataset. Có 4515 unique IDs. Family embargo, backend-context structural overlap và size probes là các estimand khác nhau. CSV giữ residual/tail/support, CPU selector, observed Graph−MLP delta và CI đúng chiều. Bootstrap 10000/seed42 conditional theo structural group; <10groups không có CI. Parent/refit ratio chỉ descriptive vì population và training procedure khác nhau.

| Loại | Case | Method | n | Groups | MAE (s) | R² | Train median MAE (s) | Fit-input collisions | Paired CI status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| family | family:qft | mali_logical_graph | 46 | 26 | 0.562994 | 0.531692 | 2.49362 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qft | mali_global_mlp | 46 | 26 | 0.512961 | 0.572636 | 2.49362 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qftentangled | mali_logical_graph | 54 | 27 | 0.74287 | 0.841393 | 2.56959 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qftentangled | mali_global_mlp | 54 | 27 | 0.551722 | 0.90718 | 2.56959 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qnn | mali_logical_graph | 26 | 14 | 36.2722 | -2197.3 | 2.65799 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qnn | mali_global_mlp | 26 | 14 | 50.7637 | -4318.14 | 2.65799 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qpeexact | mali_logical_graph | 40 | 23 | 2.20223 | -0.580058 | 3.06446 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qpeexact | mali_global_mlp | 40 | 23 | 0.603504 | 0.847131 | 3.06446 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qpeinexact | mali_logical_graph | 36 | 22 | 1.57534 | -3.97517 | 3.1813 | 0 | group_bootstrap_conditional_on_fits |
| family | family:qpeinexact | mali_global_mlp | 36 | 22 | 0.713413 | -0.0961417 | 3.1813 | 0 | group_bootstrap_conditional_on_fits |
| family | family:random | mali_logical_graph | 30 | 16 | 145.708 | -13158.5 | 3.62747 | 0 | group_bootstrap_conditional_on_fits |
| family | family:random | mali_global_mlp | 30 | 16 | 188.815 | -22124.3 | 3.62747 | 0 | group_bootstrap_conditional_on_fits |
| family | family:realamprandom | mali_logical_graph | 40 | 22 | 0.636253 | 0.426348 | 1.12082 | 0 | group_bootstrap_conditional_on_fits |
| family | family:realamprandom | mali_global_mlp | 40 | 22 | 0.951786 | -0.203254 | 1.12082 | 0 | group_bootstrap_conditional_on_fits |
| family | family:su2random | mali_logical_graph | 38 | 21 | 0.588681 | 0.540903 | 1.16792 | 0 | group_bootstrap_conditional_on_fits |
| family | family:su2random | mali_global_mlp | 38 | 21 | 1.46491 | -1.44353 | 1.16792 | 0 | group_bootstrap_conditional_on_fits |
| family | family:twolocalrandom | mali_logical_graph | 40 | 22 | 1.19371 | -0.463296 | 1.3963 | 0 | group_bootstrap_conditional_on_fits |
| family | family:twolocalrandom | mali_global_mlp | 40 | 22 | 1.25912 | -0.658131 | 1.3963 | 0 | group_bootstrap_conditional_on_fits |
| backend_context | backend_context:ibm_belem | mali_logical_graph | 24 | 3 | 0.970026 | -1.16502 | 3.84818 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_belem | mali_global_mlp | 24 | 3 | 0.492075 | 0.211479 | 3.84818 | 24 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_jakarta | mali_logical_graph | 408 | 8 | 2.06139 | -6.41667 | 0.632737 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_jakarta | mali_global_mlp | 408 | 8 | 1.24467 | -2.0619 | 0.632737 | 403 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_kyoto | mali_logical_graph | 148 | 117 | 0.664048 | 0.795814 | 2.4305 | 0 | group_bootstrap_conditional_on_fits |
| backend_context | backend_context:ibm_kyoto | mali_global_mlp | 148 | 117 | 0.541317 | 0.853584 | 2.4305 | 148 | group_bootstrap_conditional_on_fits |
| backend_context | backend_context:ibm_lagos | mali_logical_graph | 368 | 5 | 0.751983 | -2.40615 | 0.730627 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_lagos | mali_global_mlp | 368 | 5 | 0.287382 | 0.00352511 | 0.730627 | 368 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_lima | mali_logical_graph | 41 | 4 | 1.70512 | -4.74979 | 4.35656 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_lima | mali_global_mlp | 41 | 4 | 0.681366 | -0.33173 | 4.35656 | 36 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_manila | mali_logical_graph | 367 | 8 | 0.587512 | 0.578644 | 0.657305 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_manila | mali_global_mlp | 367 | 8 | 0.831318 | 0.495005 | 0.657305 | 360 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_nairobi | mali_logical_graph | 1565 | 9 | 1.48363 | -21.6575 | 1.2821 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_nairobi | mali_global_mlp | 1565 | 9 | 1.07348 | -10.0252 | 1.2821 | 1062 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_osaka | mali_logical_graph | 192 | 148 | 0.796944 | 0.733889 | 2.26474 | 0 | group_bootstrap_conditional_on_fits |
| backend_context | backend_context:ibm_osaka | mali_global_mlp | 192 | 148 | 0.430253 | 0.914454 | 2.26474 | 149 | group_bootstrap_conditional_on_fits |
| backend_context | backend_context:ibm_perth | mali_logical_graph | 351 | 9 | 1.98264 | -22.1968 | 2.18967 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_perth | mali_global_mlp | 351 | 9 | 1.49866 | -12.1608 | 2.18967 | 336 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_quito | mali_logical_graph | 1051 | 7 | 1.95833 | -6.12485 | 1.96879 | 0 | descriptive_fewer_than_10_groups |
| backend_context | backend_context:ibm_quito | mali_global_mlp | 1051 | 7 | 1.51674 | -3.55937 | 1.96879 | 1051 | descriptive_fewer_than_10_groups |
| qpack_size | qpack_size:train_through_4 | mali_logical_graph | 1321 | 3 | 0.727175 | -0.0724722 | 0.845145 | 0 | descriptive_fewer_than_10_groups |
| qpack_size | qpack_size:train_through_4 | mali_global_mlp | 1321 | 3 | 0.641164 | 0.0627129 | 0.845145 | 0 | descriptive_fewer_than_10_groups |
| qpack_size | qpack_size:train_through_5 | mali_logical_graph | 498 | 2 | 0.474422 | -4.68543 | 0.867014 | 0 | descriptive_fewer_than_10_groups |
| qpack_size | qpack_size:train_through_5 | mali_global_mlp | 498 | 2 | 0.788557 | -13.0604 | 0.867014 | 0 | descriptive_fewer_than_10_groups |
