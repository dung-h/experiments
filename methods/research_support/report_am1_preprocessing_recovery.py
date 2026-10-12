"""Derive Phase 3 reports from preprocessing receipts; never execute models."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from recover_am1_preprocessing import (ROOT, OUT, PHASE2, SCRIPTS, PANEL,
    HISTORICAL_PYTHON, CUDA_PYTHON, read, rows, sha, stable, verify_pins, write_json, write_csv)


def static_matrix_checks(old, historical):
    """Multiply constant 8x8 gate matrices, not quantum circuit/state execution."""
    import numpy as np

    def matrix(value):
        return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

    def product(steps):
        total = np.eye(8, dtype=np.complex128)
        for step in steps:
            local = matrix(step)
            qubits = step["qubits"]
            mask = sum(1 << q for q in qubits)
            expanded = np.zeros((8, 8), dtype=np.complex128)
            for i in range(8):
                for j in range(8):
                    if i & ~mask == j & ~mask:
                        a = sum(((i >> q) & 1) << b for b,q in enumerate(qubits))
                        b = sum(((j >> q) & 1) << b for b,q in enumerate(qubits))
                        expanded[i,j] = local[a,b]
            total = expanded @ total
        return total

    original = matrix(historical["static_rccx_matrix"])
    pold, phist = (product(d["static_rccx_definition"]) for d in (old, historical))
    errors = dict(old_definition_vs_static_rccx=float(np.max(np.abs(pold - original))),
                  restored_definition_vs_static_rccx=float(np.max(np.abs(phist - original))),
                  old_vs_restored_definition=float(np.max(np.abs(pold - phist))))
    ccx = np.eye(8, dtype=np.complex128)
    ccx[[3,7],:] = ccx[[7,3],:]
    errors["negative_control_CCX_difference"] = float(np.max(np.abs(original - ccx)))
    threshold = read(OUT / "acceptance_contract.json")["RCCX_static_matrix_atol"]
    return dict(status="PASS" if all(v <= threshold for k,v in errors.items() if k != "negative_control_CCX_difference")
                and errors["negative_control_CCX_difference"] > threshold else "FAIL", atol=threshold, rtol=0,
                errors=errors, positive_control="both full 8x8 decompositions vs static RCCX gate matrix",
                negative_control="CCX has different relative phases, must be rejected",
                no_QCEC_statevector_or_quantum_workload=True)


def accounting():
    timing = rows(PHASE2 / "raw_stage_timings.csv")
    selected = [r for r in timing if r["context"] == "qualified_qpu_4515" and r["method"] == "mali_global_mlp"]
    groups = {}
    for r in selected:
        key = (r["id"], int(r["repeat"]))
        bucket = groups.setdefault(key, {})
        if r["stage"] in bucket:
            raise RuntimeError("duplicate_timing_stage")
        bucket[r["stage"]] = int(r["elapsed_ns"])
    stages = ("parse_QASM", "extract_51_globals", "frozen_transform_and_tensor_encode", "batch_and_H2D")
    if len(groups) != 1350 or len({k[0] for k in groups}) != 45:
        raise RuntimeError("MLP_request_denominator")
    numerator = sum(sum(v[s] for s in stages) for v in groups.values())
    denominator = sum(v["complete_B2"] for v in groups.values())
    if any(set(v) != set(stages) | {"three_seed_inference_inverse_and_D2H", "host_median_aggregation", "complete_B2"} for v in groups.values()):
        raise RuntimeError("incomplete_MLP_stage_set")
    ratio = numerator / denominator
    prior = read(PHASE2 / "provenance_manifest.json")["validation"]["preparation_fraction_same_requests"]
    if ratio != prior:
        raise RuntimeError("preparation_share_discrepancy")
    return dict(status="CONFIRMED", requests=45, repetitions=30, paired_request_repetitions=1350,
        numerator_nanoseconds=numerator, denominator_nanoseconds=denominator, fraction=ratio,
        numerator_stages=list(stages), aggregation_order="sum stages within exact ID/repeat, then sum all1350, divide by sum independently timed total",
        includes_H2D=True, includes_parsing_and_globals=True, includes_model_load_or_Graph_preparation=False,
        denominator_includes_dispatch_and_instrumentation=True)


def build():
    import numpy as np
    pins = verify_pins()
    old, hist = (read(OUT / f"diagnostic_{f}.json") for f in ("phase2", "historical"))
    bridge = read(OUT / "bridge_validation.json")
    previous = {r["id"]:r for r in old["qpu"]}
    repaired = {r["id"]:r for r in hist["qpu"]}
    selected = rows(PHASE2 / "selected_workloads.csv")
    declared = {r["id"] for r in selected if r["context"] == "qualified_qpu_4515" and r["raw_path"]}
    if set(previous) != declared or set(repaired) != declared or len(declared) != 45:
        raise RuntimeError("cohort_changed")
    past = {}
    for p in (PHASE2 / "workers").glob("*_replay.json"):
        context = p.name.split("_fold")[0]
        for r in read(p)["histories"]:
            past[(context, r["id"], r["method"])] = r
    bridge_by_key = {(r["context"],r["id"],r["model"]):r for r in bridge["receipts"]}
    inventory, validations = [], []
    causal_equivalent = 0
    for rid in [r["id"] for r in selected if r["id"] in declared]:
        a, b = previous[rid], repaired[rid]
        if a["raw_summary"]["sequence_hash"] != b["raw_summary"]["sequence_hash"]:
            raise RuntimeError("earlier_parser_instruction_difference:" + rid)
        # Register positions and the dependency graph are inspected independently.
        registers_equal = a["raw_summary"]["registers"] == b["raw_summary"]["registers"]
        relabel = a["canonical_graph_hash"] == b["canonical_graph_hash"]
        causal_equivalent += int(relabel)
        mismatch = bool(a["mismatched_fields"])
        earliest = "RemoveBarriers pass circuit reconstruction (dag_to_circuit topological tie ordering)" if a["barrier_sequence_hash"] != b["barrier_sequence_hash"] else "none"
        old_status = past[("qualified_qpu_4515", rid, "mali_logical_graph")]["status"]
        if (old_status == "FAIL") != mismatch:
            raise RuntimeError("Phase2_failure_identity_drift")
        item = dict(id=rid, source=a["source"], family=a["family"], fold=a["fold"], backend=a["backend"],
            logical_tier=a["logical_tier"], qubits=a["raw_summary"]["q"], clbits=a["raw_summary"]["c"],
            raw_path=a["raw_path"], raw_sha256=a["raw_sha256"], snapshot_id=a["snapshot_id"],
            properties_sha256=a["properties_sha256"], frozen_graph_path=a["frozen_graph_path"], frozen_graph_sha256=a["frozen_graph_sha256"],
            original_code="prepare_logical_benchmark.py -> mali_full_features.compact_graph_record", phase2_code="run_am1_phase2.py pipeline",
            historical_environment="Qiskit2.5.2 NumPy2.2.6", phase2_environment="Qiskit0.44.2/Terra0.25.2 NumPy1.26.4",
            raw_sequence_sha256=a["raw_summary"]["sequence_hash"], register_indices_equal=registers_equal,
            instruction_definitions_identical=a["raw_summary"]["definitions"] == b["raw_summary"]["definitions"],
            earlier_unconsumed_library_metadata_difference="instruction class/definition descriptors differ; not consumed by this Graph schema",
            first_divergence=earliest, mismatch_category="B reordered graph plus node_index positional-feature incompatibility" if mismatch else "none",
            mismatched_fields=";".join(a["mismatched_fields"]), canonical_graph_equivalent=relabel,
            phase2_graph_content_sha256=a["sparse_fresh_hash"], restored_graph_content_sha256=b["sparse_fresh_hash"], frozen_graph_content_sha256=b["sparse_frozen_hash"],
            globals51_match=a["global51_match"], sparse_edge_features="none; integer adjacency only",
            schema="178 node,51 globals before frozen retained-feature mask", node_count=b["metadata"]["node_count"], edge_count=b["metadata"]["edge_count"])
        inventory.append(item)
        receipt = bridge_by_key[("qualified_qpu_4515",rid,"Graph")]
        validations.append(dict(id=rid, source=b["source"], fold=b["fold"],
            before_status="MISMATCH" if mismatch else "EXACT MATCH", status="EXACT MATCH" if not b["mismatched_fields"] and receipt["exact_model_ready_match"] else "MISMATCH",
            raw_sha256=b["raw_sha256"], frozen_sparse_sha256=b["frozen_graph_sha256"], discrete_sparse_exact=not bool(b["mismatched_fields"]),
            model_ready_exact=receipt["exact_model_ready_match"], actual_model_ready_hash=receipt["actual_hash"], frozen_model_ready_hash=receipt["expected_hash"],
            transformed_shape_dtype=json.dumps(receipt["shape_dtype"],sort_keys=True), canonical_graph_equivalent=relabel,
            phase2_order_model_compatible=False if mismatch else True))
    failids = {r["id"] for r in inventory if r["mismatched_fields"]}
    maliids = {r["id"] for r in inventory if r["source"] == "mali_real_qpu"}
    if failids != maliids or len(failids) != 38:
        raise RuntimeError("source_correlation_not_exact")
    dense = []
    olddense = {(r["context"],r["id"]):r for r in old["dense"]}
    for r in hist["dense"]:
        key = r["context"],r["id"]
        a = olddense[key]
        graph = bridge_by_key[(*key,"Graph")]
        mlp = bridge_by_key[(*key,"MLP")]
        dense.append(dict(context=r["context"], id=r["id"], family=r["family"], fold=r["fold"],
            raw_path=r["raw_path"], raw_sha256=r["raw_sha256"], before_status=a["status"], status=r["status"],
            previous_reason=a.get("reason",""), raw_rccx_declaration=r["raw_rccx_declaration"], rccx_lowered_count=r["rccx_lowered_count"],
            raw_features_DAG_exact=r["consumed_hash"] == r["frozen_consumed_hash"],
            graph_model_ready_exact=graph["exact_model_ready_match"], mlp_model_ready_exact=mlp["exact_model_ready_match"],
            graph_actual_hash=graph["actual_hash"], graph_frozen_hash=graph["expected_hash"],
            mlp_actual_hash=mlp["actual_hash"], mlp_frozen_hash=mlp["expected_hash"]))
    statics = static_matrix_checks(old, hist)
    acc = accounting()
    tests = read(OUT / "targeted_tests.json") if (OUT / "targeted_tests.json").exists() else None
    checks = dict(frozen_files_unchanged=True, frozen_file_pins=pins, QPU_accounted=45,
        QPU_before_exact=7, QPU_after_exact=sum(r["status"] == "EXACT MATCH" for r in validations),
        QPU_source_failure_identity_exact=True, canonical_graph_equivalent=causal_equivalent,
        dense_context_attempts=dict(Counter(r["context"] for r in dense)),
        dense_context_passes=dict(Counter(r["context"] for r in dense if r["status"] == "EXACT MATCH")),
        independently_encoded_model_ready_comparisons=len(bridge["receipts"]),
        independently_encoded_model_ready_exact=sum(r["exact_model_ready_match"] for r in bridge["receipts"]),
        static_RCCX=statics, MLP_accounting=acc, tests=tests,
        training_inference_latency_quantum_workloads=0, dependency_changes=False, commit_push=False,
        unchanged_split_checkpoint_transform_prediction_protocol_QASM=True,
        QPack_raw_unavailable_preserved=5,
        diagnostic_concurrency_note="Two read-only diagnostic subprocesses overlapped; per-process8GiB cap. Bridge used one preprocessing server. No measurement interference. Future measurement remains serial.",
        bridge_CUDA_calls_field_definition="0 explicit CUDA calls in harness, not a process-wide interception count. Torch __config__ import attempted device discovery and warned under address-space cap; tensors remained CPU. No forward/kernel workload was requested.",
        readiness="READY FOR B2 MEASUREMENT via declared dual-interpreter preprocessing bridge; not single-interpreter compatibility")
    if checks["QPU_after_exact"] != 45 or checks["independently_encoded_model_ready_exact"] != 210 or statics["status"] != "PASS":
        raise RuntimeError("incomplete_context_fidelity")
    documents = documents_for(checks, old, hist)
    return inventory, validations, dense, checks, documents


def documents_for(c, old, hist):
    matrix = c["static_RCCX"]["errors"]
    acc = c["MLP_accounting"]
    root = "Independent receipts: diagnostic_phase2.json, diagnostic_historical.json, bridge_validation.json. Exact inputs/producers/predictions/checkpoints/transforms pinned in acceptance_contract.json."
    docs = {
      "README.md": f"""# AM1 preprocessing recovery

RECOVERED on all declared raw-available inputs. QPU Graph:7/45 exact before,45/45 after. Dense FP32 and FP64:30/30 each, Graph and MLP. Independent frozen model-ready comparisons:210/210 exact. No new inference or latency measurement.

The repair restores the existing Qiskit2.5.2/NumPy2.2.6 preprocessing interpreter rather than sorting individual graphs to frozen expectations. The isolated stdio adapter transfers typed arrays to Torch2.7.1+cu128/NumPy1.26.4. This bridge is executable and input-compatible, but its future end-to-end latency remains unmeasured. A single-process CUDA environment with Qiskit2.5.2 has not been established.

QPack5 raw-input requests remain unavailable (45/50 QPU scope); no recipes generated. Strong classical fitted states remain missing. No complete Pareto frontier or full working RQ4 closure is claimed.

Read root_cause_analysis.md, b2_readiness_decision.md and next_measurement_amendment.md. {root}
""",
      "root_cause_analysis.md": f"""# Root causes

QPU: all45 raw parser instruction sequences and register indices agree across environments. Literal instruction-class/library-definition descriptors differ earlier in all45,including7 passing Qonductor cases. That metadata is not consumed by this encoder; it cannot explain the38-case failure by itself. The38 Ma–Li cases first diverge in model-consumed order during RemoveBarriers's DAG-to-circuit reconstruction. Terra0.25.2 (Qiskit distribution0.44.2) uses DAGOpNode.sort_key=str(qargs) and lexicographic topological ties. The2.5.2 pipeline produces a different deterministic order. RemoveFinalMeasurements subsequently preserves a version-specific order. The graph schema consumes that order, and its node_index column177 makes ordering model-visible.

All45 named-operation dependency graphs and wire/T1/T2 attributes agree under a demonstrated causal-node relabeling (node_index excluded only for this secondary equivalence test). This is not an assertion that the old ordered tensors are model-compatible. Exact compatibility is restored only through historical preprocessing and the frozen transforms. No graph-isomorphism-only PASS or per-observation reordering is used.

Mismatch fields: node_type,wire0/1,T1/T2_0/1,edge_index. Shapes/counts/dtypes and51 global features already match. Frozen NPZ archive bytes are not regenerated; content arrays are compared exactly. Edge features do not exist in this schema. Nominal calibration values, unit conversion, active-wire pruning, vocabulary, sparse deduplication and global schema remain unchanged.

Dense: the old library RCCX definition is U2(0,pi),U1(+/-pi/4),CX. The restored library definition is H,T,TDG,CX. The existing lowering correctly refuses the former unvalidated basis. This is a library-definition divergence before adapter lowering, not merely a wrong circuit name or shots/precision effect. Restored historical dispatch plus environment reproduces consumed DAG/globals exactly.

No model prediction or runtime target was used to choose the fix. {root}
""",
      "qpu_graph_stage_diffs.md": f"""# QPU stage reconciliation

Raw QASM text: pinned unchanged. Parser:45/45 ordered instructions equal. Registers/global bit positions:45/45 equal. Gate definitions may serialize/decompose differently between libraries; detailed signatures are retained in each diagnostic. No new semantic gate replacement is applied. Transpilation:absent in both paths.

First order divergence:RemoveBarriers on38/45 (all selected Ma–Li). Pruned circuit/DAG order differs on those38; same named-operation dependency topology proven by causal-wire keys and canonical sparse metadata/adjacency. Nodes/edges/sparse fields then inherit this permutation. Integer node_index stays arange, not permutation-invariant; old reordered input is therefore NOT accepted for the trained model.

178-column encoding:input/output prefix then op order; wire slots46 onward, T1/T2 at173..176, node_index177. 51 globals:gate counts44 plus7 structural metrics. Frozen retained indices/masks/scalers unchanged. Final arrays were regenerated from raw QASM, transported across interpreters, and compared against original _make_example using independent NPZ/CSV evidence. All45 Graph and45 MLP inputs exact including float32 values,shape,edge order,int64 dtype. No numeric tolerance needed for transformed inputs.

Per-ID deltas are in qpu_graph_mismatch_inventory.csv and graph_replay_validation.csv. {root}
""",
      "historical_environment_provenance.md": f"""# Environment provenance

QPU producing manifest artifacts/real_qpu/logical_inputs/feature_manifest.json explicitly records Python3.10.21,Qiskit2.5.2,NumPy2.2.6 and logical_recipe_pretranspile stage. This is recorded historical evidence, not an inference from today's packages. Current prepare_logical_benchmark.py matches the recorded preparation_script_sha2568124d799…,but the executed materializer field recordsff3a1927…; do not claim current wrapper is byte-identical to that execution snapshot. Its input_hashes[mali_full_features.py] matches current7aacb91d…; actual library bytes remain pinned before/after. Independent frozen arrays,not a wrapper-version guess,establish recovery. Upstream feature formula revision32c392a6ece276f1ff046d4e30052d0571ff6dc6 remains the original dependency.

Restored interpreter:{HISTORICAL_PYTHON}. Current package identity:{json.dumps(hist['environment'],sort_keys=True)}.

Phase2 inference interpreter:{CUDA_PYTHON}. Diagnostic identity:{json.dumps(old['environment'],sort_keys=True)}. Distribution qiskit0.44.2 is not qiskit.__version__0.25.2 (Terra component); record both meanings correctly. Original CUDA build Torch2.7.1+cu128/PyG2.6.1 unchanged. No package installation,upgrade or downgrade.

Dense IR manifest pins producing adapter/features but omits original Python/Qiskit package versions. Exact historical dense environment identity is thus unverified in that manifest. Qiskit2.5.2 is a demonstrated compatibility environment:all consumed normalized features and DAG match independently frozen dense IR. Source-QASM headers record an older generator environment, not this project's feature materializer environment.

QPU parse dispatch:Ma–Li mf.parse_qasm, Qonductor qasm2.loads with LEGACY_CUSTOM_INSTRUCTIONS. Dense historical dispatch:QuantumCircuit.from_qasm_file, restored from_qasm_str on identical bytes. RCCX definition behavior verified in both interpreters. No target-aware transpilation or backend-basis substitution occurs. {root}
""",
      "compatibility_fixes.md": f"""# Compatibility tooling changes

New isolated module benchmark_v1/scripts/am1_preprocessing_compat.py enforces Qiskit2.5.2 and NumPy2.2.6. It reuses existing parser,global extractor,compact Graph and RCCX lowering. It reconstructs consumed arrays only, preserves ordering/dtypes and uses supplied frozen transform values. No expected graph,observation ID,label,prediction,checkpoint or cache is accessible through its serving interface. Unknown sources fail closed.

The raw-to-array --serve endpoint receives JSON with base64 QASM/properties and frozen transforms, returns base64 typed arrays. One resident worker avoids per-query process spawning in a prospective measurement; IPC/encoding must remain inside future B2 accounting. Static pinned extractor functions load once during setup; no circuit features or graphs are cached. Independent consumer verification uses original _make_example and dense encode_dag/normalized_globals, not new generated targets.

New benchmark_v1/scripts/recover_am1_preprocessing.py pins source evidence, diagnoses both environments and validates the bridge. New report_am1_preprocessing_recovery.py derives CSV/reports and checks preserved timing arithmetic. New test_am1_preprocessing_recovery.py tests environment rejection, transport,dtype/schema parity,RCCX controls and complete receipt coverage.

Historical run_am1_phase2.py,mali_full_features.py,prepare_logical_benchmark.py,cudaq_qiskit_adapter.py and trained schemas are unchanged. This is a dependency-isolated recovery, not an unverified cross-version port. {root}
""",
      "dense_rccx_investigation.md": f"""# Dense RCCX reconciliation

FP32 failures:3d37464cf8c53831d2867c0c111a83e03a820e90fa26f77d01a4a85f4f9d6875 (36 RCCX),5042c6dd513b1366c0a145850de9e534b0e38aabc818b994ab8c4b6cad7ad5c8 (6),84056291d9ff227b3c615cc378349fa1154f5a122bcd0eb6bc04045bbf3affa0 (16). FP64 includes only5042… among its selected30. Same hash means same gate issue, not a precision-dependent semantic difference.

Their raw QASM includes qelib1.inc and no explicit gate rccx declaration. Old RCCXGate itself has U1/U2 definition even with original from_qasm_str parser dispatch; changing dispatch alone in the old interpreter does not fix this. Restoring existing2.5.2 library provides H/T/TDG/CX, retaining the original relative-phase gate. Definition qubit order/parameters are captured in diagnostic receipts.

Full static8x8 matrix control (not prepared-state-only):{json.dumps(matrix,sort_keys=True)}; atol1e-14,rtol0 predeclared. Both old and restored decompositions match the RCCX static gate matrix; CCX negative control is rejected. No QCEC,circuit simulation,statevector or quantum workload was run.

Scope is the pinned panel's implicit library RCCX. An initial test assuming the legacy parser would preserve an arbitrary explicit same-named QASM body failed; the parser registers built-ins. The corrected control directly verifies the lowerer's rejection of an unvalidated definition. No custom raw benchmark gate was replaced. This recovery does not qualify arbitrary custom RCCX declarations for future inputs.

All30 FP32 and30 FP64 consumed DAG/globals and frozen-transform Graph/MLP arrays match exactly. Terminal measurements and global phase remain preserved by original lowerer. No RCCX-to-CCX substitute or retrospective custom gate mapping. Raw QASM and frozen IR hashes unchanged. Detailed context rows:dense_replay_validation.csv. {root}
""",
      "mlp_preparation_accounting.md": f"""# MLP preparation accounting

CONFIRMED:{100*acc['fraction']:.8f}% (reported98.13%). Numerator={acc['numerator_nanoseconds']}ns. Denominator={acc['denominator_nanoseconds']}ns. Same45 requests×30 repetitions=1350 exact ID/repeat units.

Numerator includes parsing,51-global extraction,frozen transform/tensor encoding and batch/H2D. Sum stages within each ID/repeat, then sum across1350 and divide by sum independently timed complete_B2. Denominator includes inference,host median,dispatcher/instrumentation residuals. Model loading and Graph-only DAG construction are excluded; MLP global extraction has its upstream structural traversals. This is pooled-time weighting,not equal-query mean of ratios.

Stage medians are summarized separately across query medians. Their sum need not equal median fullB2. Do not form this98.13% by dividing summed component medians by218.77019ms. Native scalar rows do not enter either sum. No historical timing record changed or new latency measured.

Source:artifacts/am1_phase2/v1/raw_stage_timings.csv, hash{sha(PHASE2/'raw_stage_timings.csv')}. Independent arithmetic:mlp_accounting.json.
""",
      "b2_readiness_decision.md": """# B2 readiness

| Route | Raw/context coverage | Decision |
|---|---:|---|
| QPU Graph |45/45 predeclared raw-available,45/50 original scope|READY FOR B2 MEASUREMENT using declared historical preprocessing bridge|
| QPU MLP |45/45|READY; existing Phase2 timing remains a reference,not matched to the new bridge|
| Dense FP32 Graph/MLP |30/30 each|READY using the same bridge and frozen fold contracts|
| Dense FP64 Graph/MLP |30/30 each|READY using the same bridge and frozen fold contracts|
| Single CUDA-interpreter raw preprocessing |not restored on Qiskit0.44.2|BLOCKED on that path; no dependency changes authorized/performed|

READY is input compatibility,not prediction replay or measured latency. CUDA consumer constructs CPU tensors using exact typed arrays; no forward or explicit CUDA calls. Torch's configuration import attempted device discovery and warned under the address-space safety cap; this was not a model/kernel experiment. Future inference requires fresh authorized CUDA smoke/seed replay before any timing. Remeasure both Graph and MLP with identical bridge/framework/setup policy. Old MLP timing must not be spliced into a new Graph comparison.

Remaining limits:QPack5 no serialized raw inputs,competitive classical states missing,dense original preprocessing package identity incomplete,IPC/dependency split requires explicit B2 boundary,external input acquisition and cold-start remain separate. No full Pareto frontier or universal scaling claim.
""",
      "next_measurement_amendment.md": """# Proposed next measurement (not executed or authorized here)

Hypothesis:restored full raw preparation changes Graph/MLP total estimator cost relative to cached-input B1; compare accuracy and cost on identical frozen requests,not a new benchmark fit. Priority:QPU same45 IDs in Phase2,all retained38 Ma–Li and7 Qonductor; excluded QPack5 remain in50 attempt denominator. Optional separately gated dense30/context uses exact Phase2 IDs/folds,seeds andprecisioncontext.

Input boundary:resident raw logical/source QASM bytes and metadata to host prediction. Both methods route through one warm resident Qiskit2.5.2/NumPy2.2.6 CPU worker into Torch2.7.1+cu128/PyG2.6.1 CUDA worker. Pipeline includes serialization,IPC,decode,CPU preprocessing,frozen transforms,H2D,three sequential seeds,inverse transform,D2H,host median. MLP omits Graph-specific extraction. No circuit features cache. Model loading/import/input retrieval/fold routing measured or disclosed separately,not insidewarmB2.

Before timing:verify all independent raw/model-ready pins again,then authorized per-seed frozen prediction replay with original atol1e-6/rtol1e-5. Require context-wide gate. No successful subset selection. Frozen accuracy targets/predictions unchanged. No training or target simulation.

Methodology:batch1,3 fullpipeline warmups,30 repeats per request/method,rotate Graph/MLP order,CUDA synchronization around required boundaries,independent total plusstage clocks includingIPC. Per-query medians/p95; paired group bootstrap2000 seed20261011. Preserve alloutliers. Remeasure MLP rather than use Phase2 old-environment timing. Only measured newsetup can establish IPC/setup overhead.

Resources proposed:one GPU timing worker,one dedicated CPU preprocessing server,Torchthreads2/native1,previous affinity,combined monitored RSS≤8GiB andCUDAreserved≤8GiB,output≤256MiB,diskfloor3GiB,requesttimeout30s,setuptimeout120s,aggregate7200s. RSS is physical resident memory,not CUDA driver's mapped virtual address space. Do not copy this preprocessing-only validator's RLIMIT_AS onto a prospective GPU measurement without reviewing CUDA initialization; the import-time discovery warning is documented in targeted_tests.json. New approval must bind exactbridgecodehashes,cohort/environment and resource semantics. No automatic budget extension.

Stop on input/parity/integrity mismatch,unexpected CPU/GPU work,resourcefloor/timeouts. Preserve failedrequests rather than change decomposition/tolerances. Value:matched Graph–MLP fullpipeline cost,not strongest-classical frontier,crosshardware scaling or production scheduler validation. Existing pipeline cost remains source/environment dependent. Do not automatically start Phase4.
""",
    }
    checks = {k:v for k,v in c.items() if k not in {"static_RCCX","MLP_accounting"}}
    docs["validation_report.md"] = f"""# Validation

{json.dumps(checks,indent=2,sort_keys=True)}

Exact sparse and model-ready checks use independent frozen artifacts,not model outputs. Numeric rawglobal tolerance1e-12 only reflects decimal CSV storage;all transformed arrays matched exactly. Static RCCX fullmatrix evidence is narrowly about the declared gate definition,not historical logical recovery.

Commands:recover_am1_preprocessing.py --stage prepare; --stage diagnose --flavor historical in historical Python; --flavor phase2 in CUDA Python; --stage bridge in CUDA Python with CUDA_VISIBLE_DEVICES empty; --stage report/--stage check. Targeted pytest receipt is recorded separately; do not substitute old Phase2 PASS. gitdiff--check is recorded with the test receipt.

{root}
"""
    return docs


def report(check=False):
    inventory, graph, dense, checks, docs = build()
    if check:
        for name,data in [("qpu_graph_mismatch_inventory.csv",inventory),("graph_replay_validation.csv",graph),("dense_replay_validation.csv",dense)]:
            # CSV is a plain machine-readable table; compare normalized serialization.
            if rows(OUT/name) != [{k:str(v) for k,v in r.items()} for r in data]:
                raise RuntimeError("derived_table_drift:" + name)
        if read(OUT / "validation.json") != checks:
            raise RuntimeError("validation_receipt_drift")
        manifest = read(OUT/"provenance_manifest.json")
        for p,d in manifest["payload"].items():
            if sha(p) != d:
                raise RuntimeError("derived_payload_drift:" + p)
        for p,d in manifest["code_pins"].items():
            if sha(p) != d:
                raise RuntimeError("compatibility_code_drift:" + p)
        print(json.dumps(dict(status="PASS", frozen_pins=verify_pins(), QPU_exact=45,dense_exact={"fp32":30,"fp64":30},model_ready_exact=210)))
        return
    write_csv("qpu_graph_mismatch_inventory.csv",inventory)
    write_csv("graph_replay_validation.csv",graph)
    write_csv("dense_replay_validation.csv",dense)
    write_json("static_rccx_validation.json",checks["static_RCCX"])
    write_json("mlp_accounting.json",checks["MLP_accounting"])
    write_json("validation.json",checks)
    for name,text in docs.items():
        # Authored prose is emitted as an apply_patch input, not written by shell.
        print("*** Begin Patch\n*** Add File: " + str(OUT / name))
        print("\n".join("+" + line for line in text.rstrip().splitlines()))
        print("*** End Patch")
    write_json("provenance_manifest.json",dict(
        phase="AM1 Phase3 preprocessing only", authorization=read(OUT/"acceptance_contract.json")["authorization"],
        frozen_pin_count=verify_pins(), code_pins={str(p):sha(p) for p in
            (SCRIPTS/"am1_preprocessing_compat.py",SCRIPTS/"recover_am1_preprocessing.py",Path(__file__),ROOT/"benchmark_v1/tests/test_am1_preprocessing_recovery.py")},
        payload={str(p):sha(p) for p in OUT.iterdir() if p.is_file()} | {
            str(OUT/name):__import__('hashlib').sha256((text.rstrip()+"\n").encode()).hexdigest() for name,text in docs.items()},
        frozen_model_inference_or_timing=False))
