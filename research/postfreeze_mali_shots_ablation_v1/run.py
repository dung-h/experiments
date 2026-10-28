#!/usr/bin/env python3
"""Run the pre-registered Ma–Li shots-input ablation on frozen Track A."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "artifacts/postfreeze_mali_shots_ablation_v1"
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import run_qualified_logical_benchmark as runner
import run_mali_full_features as base
import run_mali_batched as batching
from shots_core import append_shots, expand_global_projection, fit_shot_transform, transform_shots, initialized_outputs_equal, verify_shots_gradient

TREATMENTS = {"M1": "mali_global_mlp_shots", "G1": "mali_logical_graph_shots"}
INTERNAL = {"M1": "mali_full_features_mlp", "G1": "mali_full_features_graph"}
SEEDS = (42, 1234, 31415)
EPOCHS = 500
NODE_BUDGET = 250000
CACHE_GIB = 8
PROTOCOL = OUT / "protocol.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, values: list[dict]) -> None:
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader(); w.writerows(values)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_protocol() -> dict:
    if not PROTOCOL.exists():
        raise RuntimeError("protocol_not_frozen_run_preflight_first")
    p = json.loads(PROTOCOL.read_text())
    if p.get("status") != "LOCKED_PRE_OUTCOME":
        raise RuntimeError("protocol_not_locked")
    if p.get("epochs") != EPOCHS or p.get("outer_folds") != list(range(5)) or p.get("seeds") != list(SEEDS):
        raise RuntimeError("protocol_runtime_constant_mismatch")
    return p


def context_and_parts(fold: int):
    ctx = runner.load_context()
    parts = runner.partitions(ctx, fold)
    return ctx, parts


def fit_transform(ctx, parts, fold):
    raw = runner.neural_transform(ctx, parts)
    shots = [float(ctx["panel_by_id"][i]["shots"]) for i in parts["fit"]]
    return {"global": raw["global_transform"], "node": raw["node_transform"],
            "shots": fit_shot_transform(shots), "fit_ids_sha256": runner.idhash(parts["fit"]),
            "fold": fold}


class ShotsCache:
    """Cache transformed input per observation, preserving observation-level shots."""
    def __init__(self, ctx, transform, budget_bytes):
        from collections import OrderedDict
        self.context, self.transform, self.budget = ctx, transform, budget_bytes
        self.items, self.bytes, self.verified = OrderedDict(), 0, set()
        self.hits, self.misses = 0, 0

    def get(self, cid, variant):
        if cid in self.items:
            self.hits += 1
            value, size = self.items.pop(cid); self.items[cid] = (value, size)
            return value
        self.misses += 1
        value = base._make_example(cid, variant, self.context["ledger"], None,
            {"global": self.transform["global"], "node": self.transform["node"]},
            self.context["feature_root"], "cpu", False, self.verified)
        shot = transform_shots(float(self.context["panel_by_id"][cid]["shots"]), self.transform["shots"])
        value = append_shots(value, shot)
        size = sum(t.numel() * t.element_size() for _, t in value if hasattr(t, "numel"))
        if size <= self.budget:
            while self.items and self.bytes + size > self.budget:
                _, (_, old_size) = self.items.popitem(last=False); self.bytes -= old_size
            self.items[cid] = (value, size); self.bytes += size
        return value


def raw_context():
    ctx = runner.load_context()
    ctx["panel_by_id"] = {r["canonical_observation_id"]: r for r in ctx["panel"]}
    if len(ctx["panel_by_id"]) != 4515:
        raise RuntimeError("frozen_panel_identity_count_changed")
    return ctx


def make_shots_model(treatment, transform, seed, torch, run_parity=False):
    base.seed_all(seed)
    model_class, model_sha = base.load_pinned_model()
    variant = INTERNAL[treatment]
    old_model = base.make_model(model_class, variant, len(transform["global"]["retained_indices"]),
                                checkpoint_layers=(treatment == "G1"))
    if treatment == "G1":
        old_model.mask = old_model.mask.to("cuda:0")
    # Save constructor RNG, then make the sole architecture change. Restore RNG so
    # dropout/training randomness follows the same stream as the frozen model.
    constructor_rng = base.capture_rng_state()
    if run_parity:
        parity_model = copy.deepcopy(old_model)
    else:
        parity_model = None
    model = old_model
    projection = expand_global_projection(model, torch)
    base.restore_rng_state(constructor_rng)
    model.to("cuda:0")
    return parity_model, model, model_sha, projection


def check_initialization(ctx, parts, transform, treatment, seed, torch):
    graph = treatment == "G1"
    old, new, model_sha, projection = make_shots_model(treatment, transform, seed, torch, True)
    old.to("cuda:0"); new.to("cuda:0")
    cache = ShotsCache(ctx, transform, 256 * 1024**2)
    by_shots = sorted(parts["fit"], key=lambda i: float(ctx["panel_by_id"][i]["shots"]))
    ids = list(dict.fromkeys([by_shots[0], by_shots[len(by_shots)//3], by_shots[2*len(by_shots)//3], by_shots[-1]]))
    old_ex, new_ex = [], []
    for cid in ids:
        ex0 = base._make_example(cid, INTERNAL[treatment], ctx["ledger"], None,
            {"global": transform["global"], "node": transform["node"]}, ctx["feature_root"], "cpu", False, cache.verified)
        ex1 = append_shots(ex0, transform_shots(float(ctx["panel_by_id"][cid]["shots"]), transform["shots"]))
        old_ex.append(ex0); new_ex.append(ex1)
    maxdiff=0.0;gradients=[]
    new.train();new.zero_grad(set_to_none=True)
    # Keep graph fixtures one at a time so this diagnostic cannot create a
    # multi-graph memory spike. Inputs and synthetic targets use fit IDs only.
    for old_one,new_one in zip(old_ex,new_ex):
        old_batch=base._batch_to_device([old_one],"cuda:0",graph)
        new_batch=base._batch_to_device([new_one],"cuda:0",graph)
        maxdiff=max(maxdiff,initialized_outputs_equal(old,new,old_batch,new_batch,torch))
        new.train()
        out=new(new_batch).reshape(-1)
        finite_targets=out.detach()+torch.tensor([0.5],device="cuda:0")
        loss=torch.nn.functional.mse_loss(out,finite_targets)
        loss.backward()
        gradients.append(float(new.gf_linear1.weight.grad[:,-1].abs().max().detach().cpu()))
        del old_batch,new_batch,out,loss
    grad_norm=max(gradients,default=0.0)
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"shots_column_gradient_gate_failed:{treatment}:{grad_norm}")
    old_count = sum(p.numel() for p in old.parameters())
    new_count = sum(p.numel() for p in new.parameters())
    if new_count - old_count != projection["added_trainable_parameters"]:
        raise RuntimeError("parameter_delta_does_not_match_projection")
    result = {"status":"PASS", "treatment":treatment, "seed":seed,
        "initialized_forward_max_abs_diff":maxdiff, "shots_column_gradient_max_abs":grad_norm,
        "old_parameters":old_count, "new_parameters":new_count,
        "added_parameters":new_count-old_count, "projection":projection,
        "fit_only_fixture_ids":ids, "model_source_sha256":model_sha,
        "shots_values_differ":len({ctx["panel_by_id"][i]["shots"] for i in ids}) > 1}
    if not result["shots_values_differ"]:
        raise RuntimeError("gradient_fixture_does_not_have_variable_shots")
    del old, new, old_ex, new_ex
    torch.cuda.empty_cache()
    return result


def cell_dir(treatment, fold, seed):
    return OUT / "training" / f"fold_{fold}" / TREATMENTS[treatment] / f"seed_{seed}"


def identity_for(ctx, parts, transform, treatment, fold, seed, model_sha, protocol_sha, env):
    return {"experiment":"postfreeze_mali_shots_ablation_v1", "treatment":treatment,
        "method_id":TREATMENTS[treatment], "internal_model":INTERNAL[treatment],
        "fold":fold, "seed":seed, "fit_ids_sha256":runner.idhash(parts["fit"]),
        "validation_ids_sha256":runner.idhash(parts["validation"]), "test_ids_sha256":runner.idhash(parts["test"]),
        "transform_sha256":stable(transform), "model_source_sha256":model_sha,
        "runner_sha256":sha(Path(__file__)), "shots_helpers_sha256":sha(HERE/"shots_core.py"),
        "panel_sha256":sha(runner.OUT/"panel.csv"), "outer_split_sha256":sha(runner.OUT/"outer_splits.csv"),
        "inner_split_sha256":sha(runner.OUT/"inner_splits.csv"), "target_sha256":sha(ROOT/"artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv"),
        "protocol_sha256":protocol_sha, "pins":ctx["pins"], "environment":env}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def train_cell(ctx, fold, seed, treatment, torch, protocol_sha, env, max_epochs=EPOCHS):
    parts = runner.partitions(ctx, fold)
    tf_path = OUT / "training" / f"fold_{fold}" / "transform.json"
    transform = fit_transform(ctx, parts, fold)
    if tf_path.exists():
        saved_tfm = read_json(tf_path)
        if saved_tfm != transform: raise RuntimeError(f"fold_transform_resume_mismatch:{fold}")
    else: atomic_json(tf_path, transform)
    folder = cell_dir(treatment, fold, seed); folder.mkdir(parents=True, exist_ok=True)
    gate_path=folder/"initialization_gate.json"
    if gate_path.exists():
        gate=read_json(gate_path)
        if gate.get("status")!="PASS" or gate.get("treatment")!=treatment or gate.get("seed")!=seed or gate.get("transform_sha256")!=stable(transform):
            raise RuntimeError("initialization_gate_identity_mismatch")
    else:
        gate=check_initialization(ctx,parts,transform,treatment,seed,torch)
        gate["transform_sha256"]=stable(transform);gate["fold"]=fold
        atomic_json(gate_path,gate)
    original_model, model, model_sha, projection = make_shots_model(treatment, transform, seed, torch)
    del original_model
    ident = identity_for(ctx,parts,transform,treatment,fold,seed,model_sha,protocol_sha,env)
    done = folder/"manifest.json"
    if done.exists():
        result=read_json(done)
        if result["status"]!="complete" or result["identity"]!=ident or any(sha(folder/n)!=h for n,h in result["outputs"].items()):
            raise RuntimeError(f"completed_cell_identity_or_checksum_mismatch:{treatment}:{fold}:{seed}")
        return result
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.0001)
    cache=ShotsCache(ctx,transform,CACHE_GIB*1024**3)
    graph=treatment=="G1"; internal=INTERNAL[treatment]
    start,best,best_epoch,best_state,elapsed_before=0,float("inf"),-1,None,0.0
    checkpoint=folder/"checkpoint.pt"; history_path=folder/"epoch_history.csv"
    if checkpoint.exists():
        saved=base.load_saved_checkpoint(checkpoint,ident,torch)
        model.load_state_dict(saved["model_state"]);optimizer.load_state_dict(saved["optimizer_state"])
        base.restore_rng_state(saved["rng_state"])
        start,best,best_epoch,best_state,elapsed_before=(saved["next_epoch"],saved["best_validation_mse"],saved["best_epoch"],saved["best_model_state"],saved["elapsed_seconds"])
        if start>0 and (not history_path.exists() or len(runner.rows(history_path))!=start):
            raise RuntimeError("resume_epoch_history_missing_or_incomplete")
    elif history_path.exists(): raise RuntimeError("orphan_epoch_history_without_checkpoint")
    if start>max_epochs: raise RuntimeError("checkpoint_ahead_of_requested_stage")
    fit_targets={i:ctx["targets"][i] for i in parts["fit"]}
    history=runner.rows(history_path) if history_path.exists() else []
    begin=time.monotonic()
    for epoch in range(start,max_epochs):
        tick=time.monotonic();model.train()
        order=__import__("numpy").random.default_rng(int(base.stable_hash([fold,seed,epoch])[:16],16)).permutation(len(parts["fit"]))
        ids=[parts["fit"][int(j)] for j in order]
        losses=[batching.loss_step(model,optimizer,ids[j:j+32],internal,NODE_BUDGET if graph else 0,cache,fit_targets,torch)
                for j in range(0,len(ids),32)]
        model.eval(); val_sum=0.0
        with torch.no_grad():
            for group in batching.physical_groups(parts["validation"],ctx["ledger"],NODE_BUDGET if graph else 0,graph):
                examples=[cache.get(i,internal) for i in group]
                payload=base._batch_to_device(examples,"cuda:0",graph)
                pred=base._prediction(model,payload)
                actual=torch.tensor([ctx["targets"][i] for i in group],dtype=torch.float32,device="cuda:0")
                val_sum+=float(torch.nn.functional.mse_loss(pred,actual,reduction="sum").cpu())
                del examples,payload,pred,actual
        val=val_sum/len(parts["validation"])
        if not math.isfinite(val) or not all(math.isfinite(x) for x in losses): raise RuntimeError("nonfinite_loss")
        if val<best:
            best,best_epoch,best_state=val,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.cuda.synchronize();elapsed=elapsed_before+time.monotonic()-begin
        base.save_checkpoint(checkpoint,{"identity":ident,"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "optimizer_state":optimizer.state_dict(),"rng_state":base.capture_rng_state(),"next_epoch":epoch+1,
            "best_validation_mse":best,"best_epoch":best_epoch,"best_model_state":best_state,"elapsed_seconds":elapsed})
        record={"epoch":epoch+1,"train_mse":sum(losses),"validation_mse":val,"best_validation_mse":best,
            "best_epoch":best_epoch+1,"epoch_seconds":time.monotonic()-tick,"elapsed_seconds":elapsed,
            "peak_cuda_bytes":torch.cuda.max_memory_allocated(),"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cache_bytes":cache.bytes,"cache_hits":cache.hits,"cache_misses":cache.misses}
        history.append(record);atomic_csv(history_path,history)
        atomic_json(folder/"progress.json",{"status":"training","completed_epochs":epoch+1,"requested_epochs":max_epochs,
            "identity":ident,"last":record,"updated_unix":time.time()})
        print(json.dumps({"stage":"train","treatment":treatment,"fold":fold,"seed":seed,**record},separators=(",",":")),flush=True)
    if max_epochs<EPOCHS:
        del model,optimizer,cache;torch.cuda.empty_cache()
        return {"status":"canary_checkpointed","completed_epochs":max_epochs,"identity":ident,
                "checkpoint_sha256":sha(checkpoint),"last_epoch":history[-1] if history else None}
    if best_state is None or best_epoch<0: raise RuntimeError("best_validation_state_missing")
    model.load_state_dict(best_state);model.eval();predictions=[]
    with torch.no_grad():
        for cid in parts["test"]:
            payload=base._batch_to_device([cache.get(cid,internal)],"cuda:0",graph)
            value=float(base._prediction(model,payload).cpu().reshape(-1)[0])
            if not math.isfinite(value): raise RuntimeError("nonfinite_test_prediction")
            predictions.append({**runner.attempt(ctx,cid,fold,TREATMENTS[treatment],value),"seed":seed})
            del payload
    atomic_csv(folder/"predictions.csv",predictions)
    result={"status":"complete","identity":ident,"epochs":EPOCHS,"best_epoch_zero_based":best_epoch,
        "best_validation_mse":best,"elapsed_seconds":history[-1]["elapsed_seconds"],
        "fit_rows":len(parts["fit"]),"validation_rows":len(parts["validation"]),"test_rows":len(parts["test"]),
        "parameter_count":sum(p.numel() for p in model.parameters()),"projection":projection,
        "outputs":{"checkpoint.pt":sha(checkpoint),"predictions.csv":sha(folder/"predictions.csv"),
                   "epoch_history.csv":sha(history_path)}}
    atomic_json(done,result);atomic_json(folder/"progress.json",{"status":"complete","identity":ident,"manifest_sha256":sha(done)})
    del model,optimizer,cache,best_state;torch.cuda.empty_cache()
    return result


def environment_receipt(torch, gpu):
    import numpy
    import sklearn
    import torch_geometric
    import platform
    return {"python":sys.executable,"python_version":sys.version,"platform":platform.platform(),
        "numpy":numpy.__version__,"scikit_learn":sklearn.__version__,"torch":torch.__version__,
        "torch_geometric":torch_geometric.__version__,"cuda_runtime":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0),"gpu_properties":{"total_memory":torch.cuda.get_device_properties(0).total_memory},
        "gpu_budget_contract":{"budget_gib":12,"reserve_gib":4},"cpu_count":os.cpu_count(),"thread_count":8,"interop_threads":1,
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("preflight","canary","train"),required=True)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--folds",default="0,1,2,3,4")
    ap.add_argument("--seeds",default="42,1234,31415")
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((OUT/"baseline_reuse_receipt.json").read_text())
    if receipt.get("status")!="PASS":raise RuntimeError("baseline_reuse_receipt_not_pass")
    ctx=raw_context()
    if args.stage=="preflight":
        print(json.dumps({"status":"PASS","cohort":len(ctx["panel"]),"groups":len({r["group_id"] for r in ctx["panel"]}),
            "baseline_cells":receipt["baseline_cells"],"existing_shots_rows":len(ctx["panel"])},indent=2));return
    p=load_protocol();protocol_sha=sha(PROTOCOL)
    import torch
    from gpu_lease import acquire_gpu_lease
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    lease=acquire_gpu_lease("postfreeze-mali-shots-ablation-v1")
    try:
        gpu=base.validate_gpu_budget(torch,budget_gib=12,reserve_gib=4)
        env=environment_receipt(torch,gpu)
        atomic_json(OUT/"training"/"execution_environment.json",env)
        if args.stage=="canary":
            result=train_cell(ctx,0,42,"G1",torch,protocol_sha,env,max_epochs=2)
            atomic_json(OUT/"training"/"canary_receipt.json",{"status":"PASS" if result.get("status")=="canary_checkpointed" else "FAIL",
                "identity":result.get("identity"),"completed_epochs":result.get("completed_epochs"),"last_epoch":result.get("last_epoch"),
                "checkpoint_sha256":result.get("checkpoint_sha256"),"environment":env,"no_test_scoring":True})
            print(json.dumps(result,indent=2));return
        if args.stage=="train":
            folds=[int(x) for x in args.folds.split(",")];seeds=[int(x) for x in args.seeds.split(",")]
            if folds!=list(range(5)) or seeds!=list(SEEDS):raise RuntimeError("only_locked_full_cell_matrix_allowed")
            canary=read_json(OUT/"training/canary_receipt.json")
            if not canary or canary.get("status")!="PASS":raise RuntimeError("canary_receipt_not_pass")
            for fold in folds:
                for t in ("G1","M1"):
                    for seed in seeds:
                        result=train_cell(ctx,fold,seed,t,torch,protocol_sha,env)
                        print(json.dumps({"cell_status":result["status"],"treatment":t,"fold":fold,"seed":seed},flush=True))
    finally:lease.release()


if __name__=="__main__":main()
