# Scientific review and final report contract

Review date: 2026-10-02. Baseline reviewed: commit `461cc6e` on
`presentation-review`. This is the S1–S5 handoff for final reporting. It
reviews existing results and authorizes routine table/document work. The
experiments already recorded in their protocols remain the numerical source.

## Evidence used by the report

The QPU evidence is the [current source-stratified aggregate](../artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/manifest.json)
and its [reader pack](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md).
QPU method identities come from [fidelity registry v2](../benchmark_v1/registry/method_fidelity_registry_v2.json).
The simulator predictor evidence is the [E6 aggregate](../artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json),
including its artifact-local method cards. The registry and the older
`report_finalization.json` predate E1–E6; their simulator availability entries
do not describe the new predictor runs. Keep those historical files intact
and cite E6 explicitly when preparing the current tables.

[Results](results.md) holds the numerical narrative; [presentation](presentation.md)
is the single slide-content brief. [Methodology](methodology.md) documents
the dataset construction and execution assumptions.

## S1: scientific consistency review

The quoted QPU and E6 errors, coverage and paired intervals agree with their
saved artifacts. The following corrections apply to the current reader path:

| Issue found | Required treatment |
| --- | --- |
| Slide brief still says Azizov GNN and the local Family-Aware adaptation are unrun | Show the evaluated E4/E5 adaptations and retain the separate original-paper limitations |
| Older scorecard/registry presented as covering every current result | Give QPU and E6 their own evidence links; use E6 method cards for new local predictors |
| 1,000 and 10,000 bootstrap policies appear without their scopes | QPU source/workflow summaries retain their recorded policy; E6 uses 10,000 paired hash replicates |
| E6 comparisons could be read as confirmatory significance tests | Intervals are pointwise and exploratory, conditional on fitted OOF predictions; no multiple-comparison adjustment or full retraining bootstrap |
| Reproduction command names the older branch | Current E1–E6 candidate is on `presentation-review`; distinguish local clone verification from later remote verification |
| Completed-run commands read like a fresh execution order | Mark them as historical recipes; final reporting uses saved predictions and measurements |

The report must admit the reconstruction steps: Ma–Li exact logical QASM;
Qonductor exact submitted physical QASM, with a separate 230-row recovered
logical-recipe tier; QPack six reconstructed structures without original
optimized angles or routing. QPU labels remain the archived observations.
Executed unified graph inputs mix logical Ma–Li/QPack with physical
Qonductor circuits. Nominal snapshots do not certify job-day calibration.

## S2: method × dataset comparison contract

Every QPU route is assigned the 8,767-row ledger and the frozen outer split.
The shared ledger does not make source timing boundaries identical. Report
source slices of the same unified fit; these are not separately trained
models. Within a source use exactly shared successful test IDs for paired
errors, and display full assigned coverage beside them. Pooled historical
scores remain secondary diagnostics.

| Domain / approach | Evaluated input and population | Evaluation target / method output | Fidelity and current status |
| --- | --- | --- | --- |
| QPU: Ma–Li-style hybrid graph + metadata | Mixed-stage unified DAGs; V3 8,763 and V3-large 8,766 predicted / 8,767 assigned | Archived observed execution/service seconds / predicted service seconds | Adaptation; paired source-level evaluation |
| QPU: Qonductor-style polynomial | Five common structural/shot inputs; 8,767 / 8,767 | Same archived target / predicted service seconds | Adaptation; same outer split as graph |
| QPU: Scholten nominal | 5,769 / 8,767; admissible nominal throughput subset | Archived service seconds / nominal CLOPS_v-like seconds; calibrated output listed separately | Adaptation; subset diagnostic; original historical CLOPS_v route unavailable |
| QPU: QCRE critical path | 8,766 / 8,767; frozen nominal Target | Archived service seconds / single-shot schedule or shot-scaled schedule | Analytical proxy; train-fold calibration is a separate adaptation |
| QPU: Qiskit duration API | 8,766 / 8,767; same Target | Archived service seconds / native single-shot schedule | Native API for schedule only; scaled/calibrated routes are adaptations; equivalent to QCRE under this Target |
| QPU: Hyb-HANAS component | 8,420 / 8,767 after bounded repair | Archived service seconds / effective cost, shot-linear cost or separately calibrated service seconds | Adaptation; extreme raw scale and invalid nominal inputs limit paper-level conclusions |
| Simulator: Ma–Li-style graph | Historical Aer 162 members / 150 hashes; E3 fixed-MPS source-DAG graph has 150 assigned hashes | Aer warm execution or fixed-MPS warm state, in separate panels | Local adaptations; historical Aer route is supplementary and not an E4 GNN |
| Simulator: Azizov-style source/hybrid/transpiled views | One FakeSherbrooke/Opt1 Aer context, 150 hashes; three GNN seed medians and classical view baselines | Noisy Aer warm execution / predicted warm execution seconds | Adaptations; five-fold OOF evaluated; reduced paper scope |
| Simulator: Family-Aware-inspired residual | Fixed CUDA-Q MPS FP64 bond-16, 150 assigned; 144 finite labels, 142 quality-pass | Warm state execution / predicted warm state seconds | Runtime-only adaptation with matched family-agnostic ablation; original joint threshold/runtime route unavailable |
| Simulator: Maestro | Component calibration and local context probes | Recorded component timing; no accepted held-out predictor score | Pilot failed stability; original Composer unavailable; retain terminal outcomes |
| Simulator: cuTensorNet | Same selected contraction plan; 603 successful sessions / 612 attempted | Actual contraction / native RUNTIME_EST | Native same-plan diagnostic; network build/path-search remain separate clocks |
| Simulator: Pasqal emu-MPS | 18-cell analog companion pilot, four quality failures | Analog-emulation timing with quality settings | Separate workload; coefficients unpromoted |

The historical Ma–Li 340 and Qonductor 4,482 source-native/source-local runs
belong in the appendix, not the unified table. V4 has a representation
contract and no trained score. Aer/CUDA-Q execution measurements are engines
and label sources, not additional prediction methods. Counts of variants,
individual seeds and equivalent Qiskit/QCRE routes must not be counted as
independent paper methods.

For E6, compare all Aer predictors on the same 150 hashes. For MPS display
the 144-finite runtime-only table and a separate 142-quality-pass table from
the same fitted models. Six unavailable targets remain in the 150 assigned
denominator; a prediction without a runtime label cannot be accuracy-scored.
The quality-pass slice describes this saved evaluation, not a new model
trained only on successful states. The q10–16 frontier has measurement
coverage but is outside these predictor accuracy panels.

## S3: findings and their limits

1. The unified V3-large graph has lower shared-row MAE than polynomial on
   Qonductor (1.027606 versus 1.394624 s; 4,481 shared rows) and QPack
   (0.943547 versus 1.221289 s; 3,945 rows). The Ma–Li difference
   (0.596772 versus 0.646791 s; 340 rows) has an interval crossing zero.
   The graph also receives two additional global inputs, so this does not
   isolate the contribution of graph structure.
2. QPack R² is negative for both adaptations (−0.094887 / −0.631629).
   Lower graph MAE does not establish a useful explanation of all runtime
   variation. Source-specific label boundaries and reconstructed instances
   remain potential explanations requiring further experiments.
3. Extending graph capacity recovers rows 39/40/41 but leaves about 76.3 s
   average absolute error on them. Coverage improvement does not establish
   accurate large-circuit extrapolation. Dynamic row4477 remains unavailable.
4. Qiskit and QCRE agree under the same frozen schedule Target, with raw
   single-shot MAE about 5.552107 s against service labels. This establishes
   implementation agreement on schedule, not two independent runtime wins.
   Calibrations may be compared only using their own fold-safe output rows.
5. On Aer, the transpiled GNN has MAE 0.3008 s versus source-view 0.4497 s;
   source minus transpiled has pointwise 95% interval [+0.0284,+0.3039] s.
   Transpiled GNN versus XGBoost (0.2970 s) is unresolved
   ([−0.1726,+0.1324] s for GNN minus XGBoost). This is one local context,
   with different feature contracts across views.
6. Fixed-MPS graph MAE is 0.3024 s on 144 finite labels and 0.3037 s on
   142 quality-pass labels. Its maximum quality-pass error is 25.0964 s;
   the mean alone hides a substantial tail. These scores do not transfer
   to other bond dimensions, engines or hardware without testing.
7. The family residual improves point MAE over its matched family-agnostic
   ablation by 0.0804 s, but the interval [−0.2852,+0.0804] s crosses zero.
   Incremental family benefit is unresolved. The original joint
   approximation/runtime method was not evaluated by this adaptation.

The paired intervals condition on the saved fits. Overlapping outer-training
sets, only three neural seeds, small effective circuit populations, rare
families, nominal calibration and missing outputs limit generalization.
E6's 28 pairwise intervals are unadjusted exploratory comparisons. Report
effect size and interval rather than a blanket claim that a paper wins.

## S4: presentation handoff

Use the updated [presentation brief](presentation.md): dataset construction,
representation assumptions, grouped evaluation, QPU method table, separate
Aer/MPS predictor tables, native/analog diagnostics, and bounded findings.
Each displayed number needs its population, unit, coverage and source link
in notes. The review is complete for the content contract; rendered slides
still need routine production and final inspection.

## S5: publication boundary

Author-requested review-branch sharing is already authorized. The current
review package can be committed/shared under that existing direction.
[S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md)
still governs a formally licensed public release. The repository owner must
choose the code license and supply accurate citation authors/metadata;
no such choice is inferred here.

Complete external Ma–Li and Qonductor archives stay outside the package.
Public staging may retain this exclusion; recovering them is not required
to release a table-rebuild package. Document rights/notices for materials
actually included, including derived data and upstream source snippets.
Do not vendor the external Maestro GPL checkout without reviewing its
licensing obligations. Missing QPack original execution inputs remain
unavailable and reconstructed inputs remain disclosed.

The current clone verification covers CSV restoration, inventory, table
rebuild and tests using the existing verification interpreter. Fresh
dependency installation and GPU retraining are separate unverified steps.
Before declaring a remote release reproducible, check the actual pushed
branch/commit using the published instructions.

## Routine-agent work now unlocked

| Task | Inputs / acceptance | Parallelism |
| --- | --- | --- |
| C1: reader tables | QPU pack plus E6; keep the two clocks, exact shared rows, full coverage and method cards; mark superseded availability entries historical | Can run alongside C3/C5 |
| C2: figures | C1 and the bounded findings above; attach source IDs, units and captions; at most four useful plots | After C1; alongside C4 |
| C3: documentation synchronization | Use this review and updated presentation brief; no numerical artifact rewriting or silent registry promotion | Can run alongside C1/C5 |
| C4: slides | Updated single presentation brief and verified C1 tables; render and inspect deck | After C1; alongside C2 |
| C5: release provenance inventory | List included materials, upstream notices, exclusions and unresolved owner decisions; do not guess license/authors | Can run alongside C1/C3 |
| C6: reproduction | Refresh inventory after edits, restore parts and rebuild in fresh clone; attempt documented fresh dependency install and report actual outcome | After package changes settle |
| C7: final review | Check cross-links, stale availability claims, displayed metrics and rendered slides against source tables | After C1–C6 |

Content/table work can proceed now. A licensed release depends on the owner
decisions above; no new experiments are required for this reporting queue.
