# NoTaDS and HyperQ: estimator claims versus public implementation

Read-only source audit, 2026-10-27. No upstream scripts imported/executed, no circuit execution, training, or frozen benchmark change. Public sources were read; `source_inventory.json` records retrieved source hashes and line references.

## NoTaDS

Paper: [arXiv:2309.06005v3](https://arxiv.org/html/2309.06005v3), especially sections 5.1–5.2. Its central claim concerns fidelity-aware scheduling under modeled execution-time budgets. Section 5.2 explicitly adopts abstract values t1=1 and t2=10. This is not a declaration that a gate takes one or ten seconds. Paper figures on scheduling/fidelity do not establish predictive MAE against archived provider Result.time_taken.

The paper counts layers containing only 1Q gates as kappa1 and layers containing 2Q gates as kappa2, assigning each layer its longest gate time. Its score is kappa1 + 10*kappa2. A physical-time version is permitted conceptually, but the paper warns that absolute gate times vary across hardware.

Public implementation reviewed at commit `ea1303e545bb4c94bcbda1020a6d845ada6a9506`:

- [NoTODS.py lines 41–42](https://github.com/debasmita2102/NoTODS/blob/ea1303e545bb4c94bcbda1020a6d845ada6a9506/NoTODS.py#L41): weights 10 and 1.
- [Line 154](https://github.com/debasmita2102/NoTODS/blob/ea1303e545bb4c94bcbda1020a6d845ada6a9506/NoTODS.py#L154): sum of separately filtered 2Q and 1Q depths, `10*depth(2Q)+depth(1Q)`.
- [Line 179](https://github.com/debasmita2102/NoTODS/blob/ea1303e545bb4c94bcbda1020a6d845ada6a9506/NoTODS.py#L179): multiply by the number of circuit-cutting instances for the time-budget constraint.

The two depth definitions can differ when independent 1Q and 2Q gates occupy the same original layer. This is a paper/code discrepancy to declare, not evidence of measured prediction failure. The implementation does not supply an observed-runtime seconds conversion, shot scaling or service-overhead model for the Track-A target. A calibrated seconds predictor would be a separately labeled adaptation; a backend-duration implementation would also require an explicit physical-time contract.

## HyperQ paper

Paper: [Quantum Virtual Machines, OSDI 2025](https://www.usenix.org/system/files/osdi25-tao.pdf).

- Section 3.3, printed page 418: explicitly describes critical-path length weighted by gate processing times obtained from Qiskit's backend properties. It also includes measurement/reset costs for time multiplexing. Thus the user's description of the **paper method** as a backend-duration critical-path estimator is supported.
- Section 3.4, printed page 419: describes summing gate times along the longest gate sequence in the composite circuit and submitting with a timeout of 1.5 times the estimated running time.

These are timing mechanisms inside virtualization/scheduling. The paper's end-to-end systems improvements should not be relabeled as an estimator MAE result. The exact complete conversion from circuit-path duration to multi-shot job time, including service overhead, is not recovered from the inspected public execution path.

## HyperQ public repository and history

Repository commit: `11a2e17e1204fee7c51df36415b62ce6c87b45e2`. All **20 Python files plus README** were fetched successfully and searched for duration, critical/longest paths, gate lengths, estimates, depth, timeouts and calibration use. No training or submitted job was executed.

Actual current paths:

1. [HypervisorBackend.py line 340](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/HypervisorBackend.py#L340): region height uses `qc.depth()`.
2. [Lines 393–395](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/HypervisorBackend.py#L393): time fitting uses `qc.depth()+50`; 50 is in this depth-based heuristic, not documented as 50 seconds.
3. [getdata/get_calibration.py lines 22–24](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/getdata/get_calibration.py#L22): records ECR gate error and readout error, not gate durations.
4. [CombinerJob.py line 45](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/CombinerJob.py#L45) and [getdata/get_job_time.py line 46](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/getdata/get_job_time.py#L46): retrieve measured execution-span duration after execution. This is not an estimate available before execution.
5. [README line 127](https://github.com/1640675651/HyperQ/blob/11a2e17e1204fee7c51df36415b62ce6c87b45e2/README.md#L127): documents the switch from legacy backend.run(dynamic=True) to Sampler, with changed behavior/time-scheduling problems.

The public HypervisorBackend.py history contains three file-changing commits: `c1db87cb6b92795e83be7af4b4173cda86254650` (2025-10-15), `d9c6e7d87ba48a98d9f9a7892c82cee7758a51ea` (2025-05-25), and `09fc8041616354f45f445859f5f1f6e4141a67bd` (2025-04-13). The file at all three commits uses depth and depth+50; the discrepancy therefore predates the current Sampler revision. The inspected current Python tree has no implementation of the paper-described weighted-critical-path estimator or its 1.5x timeout rule. This does not establish what code the authors used outside the released history.

## Benchmark implications

| Candidate | What can be reproduced from evidence | What must remain separate |
|---|---|---|
| NoTaDS | Abstract weighted-layer paper score, or filtered-depth code score | Neither score is already a runtime prediction in seconds. Conversion/calibration must be declared. |
| HyperQ paper method | Duration-weighted critical-path calculation from its documented algorithm | A reimplementation must be labeled as such; the public scheduler currently uses depth. |
| HyperQ released scheduler | Depth-based time packing and post-execution span retrieval | Do not call execution-span retrieval a predictor, or noise calibration a runtime calibration. |

HyperQ's paper estimator is conceptually related to existing QCRE/Qiskit duration routes, but exact equivalence has not been checked under identical instruction handling and inputs. Do not declare it a duplicate solely from the description. No new MAE or baseline is computed in this audit.

## Correction to the preceding conversational answer

The earlier answer inspected the current scheduler and left the HyperQ paper claim unresolved. The paper explicitly supports the weighted-critical-path description. Both facts must be retained: **paper says weighted critical path; released code reviewed here uses depth**. NoTaDS is an abstract time-budget model in its published setup, not a validated seconds predictor for provider-recorded job runtimes.
