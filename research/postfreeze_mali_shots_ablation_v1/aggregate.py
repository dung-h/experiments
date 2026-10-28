#!/usr/bin/env python3
"""Aggregate saved predictions only; never fits or runs an estimator."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import r2_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/"artifacts/postfreeze_mali_shots_ablation_v1"
sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"));import run_qualified_logical_benchmark as runner
TREAT={"M0":"mali_global_mlp","M1":"mali_global_mlp_shots","G0":"mali_logical_graph","G1":"mali_logical_graph_shots"}
BASE={"M0":"mali_global_mlp","G0":"mali_logical_graph"};NEW={"M1":"mali_global_mlp_shots","G1":"mali_logical_graph_shots"}
SEEDS=(42,1234,31415)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def write(p,rows):
    if not rows:return
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(p).with_name(Path(p).name+".partial")
    with tmp.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush()
    tmp.replace(p)
def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    if (y<0).any() or (p<=-1).any():raise RuntimeError("log1p_metric_requires_y_nonnegative_and_predictions_above_minus_one")
    return {"rows":len(y),"mae_seconds":float(np.mean(np.abs(e))),"medae_seconds":float(np.median(np.abs(e))),
        "rmse_seconds":float(np.sqrt(np.mean(e*e))),"raw_seconds_r2":float(r2_score(y,p)),
        "log1p_seconds_r2":float(r2_score(np.log1p(y),np.log1p(p))),
        "mean_signed_error_seconds":float(np.mean(e)),"p90_abs_error_seconds":float(np.quantile(np.abs(e),.9)),
        "p99_abs_error_seconds":float(np.quantile(np.abs(e),.99)),"max_abs_error_seconds":float(np.max(np.abs(e)))}
def group_maes(ids,groups,truth,pred):
    values=defaultdict(list)
    for i in ids:values[groups[i]].append(abs(pred[i]-truth[i]))
    return {g:float(np.mean(v)) for g,v in values.items()}
def main():
    receipt=json.loads((OUT/"baseline_reuse_receipt.json").read_text())
    if receipt.get("status")!="PASS":raise RuntimeError("baseline_reuse_not_pass")
    protocol=json.loads((OUT/"protocol.json").read_text())
    if protocol.get("status")!="LOCKED_PRE_OUTCOME":raise RuntimeError("protocol_not_locked")
    ctx=runner.load_context();panel={r["canonical_observation_id"]:r for r in ctx["panel"]}
    if len(panel)!=4515:raise RuntimeError("frozen_panel_changed")
    ids=sorted(panel);groups={i:panel[i]["group_id"] for i in ids};truth={i:float(ctx["targets"][i]) for i in ids}
    folds={i:int(ctx["outer"][i]["outer_fold"]) for i in ids}
    base_rows=read(OUT/"baseline_predictions.csv")
    predictions={}
    for r in base_rows:
        m=r["method"];tag={"mali_global_mlp":"M0","mali_logical_graph":"G0"}.get(m)
        if tag is None:continue
        predictions.setdefault(tag,{})[r["canonical_observation_id"]]=float(r["prediction"])
    seed_output=[]
    for tag,method in NEW.items():
        for fold in range(5):
            parts=runner.partitions(ctx,fold);expected=set(parts["test"])
            for seed in SEEDS:
                d=OUT/"training"/f"fold_{fold}"/method/f"seed_{seed}"
                manifest=json.loads((d/"manifest.json").read_text())
                if manifest.get("status")!="complete" or manifest.get("epochs")!=500:
                    raise RuntimeError(f"cell_incomplete:{tag}:{fold}:{seed}")
                if manifest["identity"]["test_ids_sha256"]!=runner.idhash(parts["test"]):raise RuntimeError("fold_test_identity_mismatch")
                for name,h in manifest["outputs"].items():
                    if sha(d/name)!=h:raise RuntimeError(f"cell_output_checksum_mismatch:{d/name}")
                rows=read(d/"predictions.csv")
                if len(rows)!=len(expected) or {r["canonical_observation_id"] for r in rows}!=expected:
                    raise RuntimeError(f"cell_prediction_coverage_mismatch:{tag}:{fold}:{seed}")
                for r in rows:
                    i=r["canonical_observation_id"];p=float(r["predicted_seconds"])
                    if float(r["actual_seconds"])!=truth[i] or int(r["outer_fold"])!=fold or int(r["seed"])!=seed or not math.isfinite(p):
                        raise RuntimeError(f"invalid_saved_oof_row:{i}")
                    seed_output.append({"canonical_observation_id":i,"group_id":groups[i],"source_id":panel[i]["source_id"],
                        "outer_fold":fold,"treatment":tag,"method_id":method,"seed":seed,"actual_seconds":truth[i],"predicted_seconds":p})
    write(OUT/"aggregate/seed_predictions.csv",seed_output)
    for tag,method in NEW.items():
        byseed=defaultdict(dict)
        for r in seed_output:
            if r["treatment"]==tag:byseed[int(r["seed"])][r["canonical_observation_id"]]=float(r["predicted_seconds"])
        if set(byseed)!=set(SEEDS) or any(set(byseed[s])!=set(ids) for s in SEEDS):raise RuntimeError("new_treatment_oof_incomplete")
        predictions[tag]={i:float(np.median([byseed[s][i] for s in SEEDS])) for i in ids}
    if any(set(predictions.get(t,{}))!=set(ids) for t in TREAT):raise RuntimeError("common_oof_population_missing")
    oof=[]
    for i in ids:
        for tag in TREAT:
            oof.append({"canonical_observation_id":i,"group_id":groups[i],"source_id":panel[i]["source_id"],
                "outer_fold":folds[i],"shots":panel[i]["shots"],"treatment":tag,"method_id":TREAT[tag],
                "actual_seconds":truth[i],"predicted_seconds":predictions[tag][i],"absolute_error_seconds":abs(predictions[tag][i]-truth[i]),
                "signed_error_seconds":predictions[tag][i]-truth[i]})
    write(OUT/"aggregate/oof_predictions.csv",oof)
    metric_rows=[];group_rows=[]
    for tag in TREAT:
        m=metric([truth[i] for i in ids],[predictions[tag][i] for i in ids])
        gmae=group_maes(ids,groups,truth,predictions[tag]);m["group_balanced_mae_seconds"]=float(np.mean(list(gmae.values())))
        src_maes=[]
        for src in sorted({panel[i]["source_id"] for i in ids}):
            si=[i for i in ids if panel[i]["source_id"]==src]
            src_maes.append(float(np.mean([abs(predictions[tag][i]-truth[i]) for i in si])))
        m["source_macro_mae_seconds"]=float(np.mean(src_maes))
        m.update(treatment=tag,method_id=TREAT[tag],unique_groups=len(gmae),coverage_rows=len(predictions[tag]))
        metric_rows.append(m)
        for g,v in gmae.items():group_rows.append({"treatment":tag,"group_id":g,"rows":sum(groups[i]==g for i in ids),"mae_seconds":v})
    write(OUT/"aggregate/metrics.csv",metric_rows);write(OUT/"aggregate/group_metrics.csv",group_rows)
    slice_rows=[]
    for tag in TREAT:
        for field in ("source_id",):
            values=sorted({panel[i][field] for i in ids})
            for value in values:
                subset=[i for i in ids if panel[i][field]==value]
                m=metric([truth[i] for i in subset],[predictions[tag][i] for i in subset])
                gmae=group_maes(subset,groups,truth,predictions[tag]);m.update(treatment=tag,slice_type=field,slice_value=value,
                    group_balanced_mae_seconds=float(np.mean(list(gmae.values()))),unique_groups=len(gmae))
                slice_rows.append(m)
        for fold in range(5):
            subset=[i for i in ids if folds[i]==fold]
            m=metric([truth[i] for i in subset],[predictions[tag][i] for i in subset])
            gmae=group_maes(subset,groups,truth,predictions[tag]);m.update(treatment=tag,slice_type="outer_fold",slice_value=fold,
                group_balanced_mae_seconds=float(np.mean(list(gmae.values()))),unique_groups=len(gmae))
            slice_rows.append(m)
    write(OUT/"aggregate/slice_metrics.csv",slice_rows)
    per_seed=[]
    for r in seed_output:
        tag=r["treatment"]
        # individual seeds are descriptive; the baseline is the frozen median ensemble.
        if tag not in ("M1","G1"):continue
    for tag in NEW:
        for seed in SEEDS:
            subset=[r for r in seed_output if r["treatment"]==tag and int(r["seed"])==seed]
            m=metric([float(r["actual_seconds"]) for r in subset],[float(r["predicted_seconds"]) for r in subset])
            per_seed.append({"treatment":tag,"seed":seed,**m})
    write(OUT/"aggregate/seed_metrics.csv",per_seed)
    # Paired group bootstrap. Each resampled unit is a complete structural group.
    ordered_groups=sorted(set(groups.values()));gindex={g:j for j,g in enumerate(ordered_groups)}
    group_errors={tag:np.array([group_maes(ids,groups,truth,predictions[tag])[g] for g in ordered_groups]) for tag in TREAT}
    rng=np.random.default_rng(42);draws=rng.integers(0,len(ordered_groups),size=(10000,len(ordered_groups)),dtype=np.int32)
    np.save(OUT/"aggregate/group_bootstrap_draws_seed42.npy",draws,allow_pickle=False)
    comparisons=[]
    for treatment,control in (("G1","G0"),("M1","M0")):
        diff=group_errors[treatment]-group_errors[control]
        point=float(diff.mean());boot=diff[draws].mean(axis=1)
        comparisons.append({"contrast":f"{treatment}-{control}","treatment":treatment,"control":control,
            "delta_group_balanced_mae_seconds":point,"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),
            "ci97_5_low":float(np.quantile(boot,.0125)),"ci97_5_high":float(np.quantile(boot,.9875)),
            "empirical_bootstrap_fraction_delta_below_zero":float(np.mean(boot<0)),"bootstrap_replicates":len(boot),
            "bootstrap_unit":"structural_group","groups":len(ordered_groups),"seed":42,
            "interpretation":"negative favors adding shots; bootstrap fraction is descriptive, not a p-value"})
    write(OUT/"aggregate/paired_comparisons.csv",comparisons)
    # Shot support and same-QASM/backend Qonductor contexts; these are OOF descriptive diagnostics.
    support=[]
    for fold in range(5):
        p=runner.partitions(ctx,fold)
        for split in ("fit","validation","test"):
            subset=[panel[i] for i in p[split]];vals=[float(r["shots"]) for r in subset]
            support.append({"fold":fold,"split":split,"rows":len(subset),"unique_shots":len(set(vals)),
                "min_shots":min(vals),"max_shots":max(vals),"fit_only_transform_mean":protocol["folds"][fold]["shots_transform"]["mean"],
                "fit_only_transform_std_ddof1":protocol["folds"][fold]["shots_transform"]["std_ddof1"]})
    write(OUT/"aggregate/shot_support_by_fold.csv",support)
    byctx=defaultdict(list)
    for i in ids:
        r=panel[i]
        if r["source_id"]=="qonductor_single_circuit_ibm" and r["source_qasm_sha256"]:
            byctx[(r["source_qasm_sha256"],r["backend"])].append(i)
    diagnostics=[]
    for (h,b),members in byctx.items():
        levels=sorted({int(float(panel[i]["shots"])) for i in members})
        if len(levels)<2:continue
        for tag in TREAT:
            for level in levels:
                chosen=[i for i in members if int(float(panel[i]["shots"]))==level]
                diagnostics.append({"source_qasm_sha256":h,"backend":b,"treatment":tag,"shots":level,"rows":len(chosen),
                    "target_median_seconds":float(np.median([truth[i] for i in chosen])),
                    "prediction_median_seconds":float(np.median([predictions[tag][i] for i in chosen])),
                    "all_predictions_oof":True,"groups":json.dumps(sorted({groups[i] for i in chosen}))})
    write(OUT/"aggregate/qonductor_matched_shot_contexts.csv",diagnostics)
    matched_contexts=sum(len({int(float(panel[i]["shots"])) for i in members})>1 for members in byctx.values())
    print(json.dumps({"status":"AGGREGATED","metric_rows":metric_rows,"primary_comparisons":comparisons,
        "oof_rows":len(oof),"new_seed_rows":len(seed_output),"matched_qonductor_hash_backend_contexts":matched_contexts},indent=2))
if __name__=="__main__":main()
