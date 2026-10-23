#!/usr/bin/env python3
"""Authorized A1 100%-QPU factorial runner.

Exactly four treatments × five frozen folds × three frozen seeds are possible.
It has no flags for sample fractions, paper-like transfer, topology controls,
simulator execution, QPU execution, sweeps, or altered splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
UPSTREAM = Path("/home/server/Documents/Quantum-Execution-Time-Prediction")
PIN = "32c392a6ece276f1ff046d4e30052d0571ff6dc6"
PYTHON = Path("/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python")
SEEDS = (42, 1234, 31415)
TREATMENTS = {
    "M-S": {"architecture": "MLP", "initialization": "SCRATCH", "cohort": "U-FULL-V3"},
    "M-P": {"architecture": "MLP", "initialization": "STRICT_PRETRAINED", "cohort": "U-FULL-V3"},
    "G-S": {"architecture": "GRAPH_MLP", "initialization": "SCRATCH", "cohort": "U-GRAPH-V3"},
    "G-P": {"architecture": "GRAPH_MLP", "initialization": "STRICT_PRETRAINED", "cohort": "U-GRAPH-V3"},
}
PANEL = REPO / "artifacts/real_qpu/qualified_logical_panel/panel.csv"
OUTER = REPO / "artifacts/real_qpu/qualified_logical_panel/outer_splits.csv"
INNER = REPO / "artifacts/real_qpu/qualified_logical_panel/inner_splits.csv"
QUALIFICATION = OUT / "checkpoint_qualification.json"
STRICT_PRETRAIN = OUT / "pretraining/valid_pinned_simple_model"
FACTORIAL = OUT / "qpu_factorial"
GRAPH_NODE_CAP = 650_000
GRAPH_EDGE_CAP = 1_100_000


def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
def scope(path: Path) -> Path:
    path=path.resolve(); root=OUT.resolve()
    if path!=root and root not in path.parents: raise RuntimeError(f"path_outside_a1_namespace:{path}")
    return path
def write_bytes(path: Path, value: bytes) -> None:
    path=scope(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+".",suffix=".partial",delete=False) as f:
        tmp=Path(f.name); f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def write_json(path: Path, value: Any) -> None: write_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]) -> None:
    out=io.StringIO(newline=""); w=csv.DictWriter(out,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows); write_bytes(path,out.getvalue().encode())
def read_csv(path: Path) -> list[dict[str,str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def seed_all(seed:int) -> None:
    import numpy as np, torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def pinned_model():
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv,global_mean_pool
    blob=subprocess.check_output(["git","-C",str(UPSTREAM),"show",f"{PIN}:model/transformer_model.py"])
    expected="d6d250b2cb3082efc756ae9c5c269d4fe40bbdf49c743c535cddf6a0691c5683"
    if sha_bytes(blob)!=expected:raise RuntimeError("pinned_model_blob_hash_mismatch")
    ns={"torch":torch,"F":F,"TransformerConv":TransformerConv,"global_mean_pool":global_mean_pool};exec(compile(blob,"<a1-pinned-model>","exec"),ns)
    return ns["Simple_Model"],{"commit":PIN,"path":"model/transformer_model.py","sha256":expected}
def args_for(architecture:str):
    return SimpleNamespace(use_graph_features=architecture=="GRAPH_MLP",use_global_features=True,use_gate_type=True,use_qubit_index=True,use_T1T2=False,use_gate_error=False,use_gate_index=True,num_layers=3)
def model_for(architecture:str):
    Model,provenance=pinned_model();m=Model(args_for(architecture),length_of_x=178,length_of_gf=51)
    if architecture=="GRAPH_MLP" and (int(m.mask.sum())!=174 or bool(m.mask[173:177].any())):raise RuntimeError("circuit_only_mask_failure")
    return m,provenance
def fp(model):
    state=model.state_dict();return {"keys":list(state),"shapes":{k:list(v.shape) for k,v in state.items()},"parameters":sum(p.numel() for p in model.parameters())}
def require_gate():
    q=json.loads(QUALIFICATION.read_text())
    if q.get("PRETRAINING_PROVENANCE")!="PASS" or q.get("CHECKPOINT_PARITY")!="PASS":raise RuntimeError("checkpoint_qualification_not_pass")
def source_checkpoint(architecture:str,seed:int)->Path:
    root=STRICT_PRETRAIN/architecture/f"seed_{seed}"; receipt=json.loads((root/"terminal_receipt.json").read_text());want=receipt["checkpoint_sha256"]
    candidates=[p for p in root.rglob("best.pt") if sha_file(p)==want]
    canonical=root/"best.pt"
    if canonical in candidates:return canonical
    if len(candidates)!=1:raise RuntimeError(f"source_checkpoint_resolution_failure:{architecture}:{seed}")
    return candidates[0]
def partitions(fold:int, panel_by:dict[str,dict[str,str]], outer:list[dict[str,str]], inner:list[dict[str,str]]):
    test={r["canonical_observation_id"] for r in outer if int(r["outer_fold"])==fold}
    indexed={r["canonical_observation_id"]:r for r in inner if int(r["outer_fold"])==fold}
    if set(indexed)!=(set(panel_by)-test):raise RuntimeError(f"inner_membership_failure:{fold}")
    valid={i for i,r in indexed.items() if int(r["inner_fold"])==0};fit={i for i,r in indexed.items() if int(r["inner_fold"]) in (1,2,3)}
    for a,b in ((fit,valid),(fit,test),(valid,test)):
        if {panel_by[i]["group_id"] for i in a}&{panel_by[i]["group_id"] for i in b}:raise RuntimeError(f"group_leakage:{fold}")
    return {"fit":sorted(fit),"validation":sorted(valid),"test":sorted(test)}
def transforms(panel_by,ids):
    import numpy as np
    v=np.asarray([[float(panel_by[i][f"g{j:02d}"]) for j in range(51)] for i in ids],dtype=np.float32);m=v.mean(0);s=v.std(0);return m,np.where(s==0,1.,s),np.flatnonzero(s==0).tolist()
def load_graph(row):
    import numpy as np
    sys.path.insert(0,str(REPO/"benchmark_v1/scripts"));from mali_full_features import expand_dense_x
    p=REPO/row["graph_file"]
    if not p.is_file() or sha_file(p)!=row["graph_sha256"]:raise RuntimeError(f"qpu_graph_hash_failure:{row['canonical_observation_id']}")
    with np.load(p,allow_pickle=False) as z:return expand_dense_x({k:z[k] for k in z.files}),z["edge_index"].copy()
def graph_transforms(panel_by,ids):
    import numpy as np
    total=np.zeros(178,dtype=np.float64);sq=total.copy();n=0
    for i in ids:
        x,_=load_graph(panel_by[i]);total+=x.sum(0);sq+=(x*x).sum(0);n+=len(x)
    m=(total/n).astype(np.float32);s=np.sqrt(np.maximum(sq/n-m.astype(np.float64)**2,0)).astype(np.float32);return m,np.where(s==0,1.,s)
def graph_item(row,gmean,gstd,nmean,nstd):
    import numpy as np,torch
    from torch_geometric.data import Data
    x,e=load_graph(row);g=(np.asarray([float(row[f"g{j:02d}"]) for j in range(51)],dtype=np.float32)-gmean)/gstd
    return Data(x=torch.as_tensor((x-nmean)/nstd),edge_index=torch.as_tensor(e,dtype=torch.long),global_features=torch.as_tensor(g).reshape(1,51),y=torch.tensor(float(row["target_seconds"]),dtype=torch.float32))
def fit_cell(treatment:str,fold:int,seed:int)->dict[str,Any]:
    import numpy as np,torch
    from torch_geometric.data import Batch
    require_gate()
    if treatment not in TREATMENTS or fold not in range(5) or seed not in SEEDS:raise RuntimeError("unauthorized_cell")
    spec=TREATMENTS[treatment];arch=spec["architecture"];d=FACTORIAL/treatment/f"fold_{fold}"/f"seed_{seed}";terminal=d/"terminal_receipt.json"
    if terminal.is_file():return json.loads(terminal.read_text())
    panel=read_csv(PANEL);panel_by={r["canonical_observation_id"]:r for r in panel};part=partitions(fold,panel_by,read_csv(OUTER),read_csv(INNER));rows={k:[panel_by[i] for i in v] for k,v in part.items()}
    seed_all(seed);scratch,prov=model_for(arch);model,_=model_for(arch);before=fp(scratch)
    if fp(model)!=before:raise RuntimeError("scratch_preload_parity_failure")
    source=None
    if spec["initialization"]=="STRICT_PRETRAINED":
        source=source_checkpoint(arch,seed);ck=torch.load(source,map_location="cpu",weights_only=False);model.load_state_dict(ck["model_state"],strict=True)
    device=torch.device("cuda");model=model.to(device);gmean,gstd,zero=transforms(panel_by,part["fit"])
    transform={"global_mean":gmean.tolist(),"global_std":gstd.tolist(),"zero_variance_global_indices":zero,"raw_width":51,"fit_id_sha256":sha_bytes("\n".join(part["fit"]).encode()),"zero_total_pruning":False}
    if arch=="MLP":
        def batch(items):
            x=np.asarray([[(float(r[f"g{j:02d}"])-gmean[j])/gstd[j] for j in range(51)] for r in items],dtype=np.float32);y=np.asarray([float(r["target_seconds"]) for r in items],dtype=np.float32);x=torch.as_tensor(x,device=device);return SimpleNamespace(x=x,edge_index=None,global_features=x,batch=None),torch.as_tensor(y,device=device)
        def predict(items):data,y=batch(items);return model(data).reshape(-1),y
        policy={"kind":"fixed_row_batch","rows":256}
    else:
        nmean,nstd=graph_transforms(panel_by,part["fit"]);transform.update({"node_mean":nmean.tolist(),"node_std":nstd.tolist(),"stored_node_width":178,"active_node_width":174})
        sizes={i:(int(panel_by[i]["node_count"]),int(panel_by[i]["edge_count"])) for i in panel_by}
        def predict(items):
            b=Batch.from_data_list([graph_item(r,gmean,gstd,nmean,nstd) for r in items]).to(device);return model(b).reshape(-1),b.y.reshape(-1)
        def graph_packs(items):
            packs=[];cur=[];nodes=edges=0
            for r in items:
                a,b=sizes[r["canonical_observation_id"]]
                if cur and(nodes+a>GRAPH_NODE_CAP or edges+b>GRAPH_EDGE_CAP):packs.append(cur);cur=[];nodes=edges=0
                cur.append(r);nodes+=a;edges+=b
            if cur:packs.append(cur)
            return packs
        policy={"kind":"fixed_node_edge_capped","node_cap":GRAPH_NODE_CAP,"edge_cap":GRAPH_EDGE_CAP}
    write_json(d/"identity_manifest.json",{"study_id":"postfreeze_track_a_transfer_v1","status":"RUNNING","treatment":treatment,"architecture":arch,"initialization":spec["initialization"],"fold":fold,"seed":seed,"epochs":500,"partitions":{k:{"count":len(v),"sha256":sha_bytes("\n".join(v).encode())} for k,v in part.items()},"source_checkpoint_sha256":sha_file(source) if source else None,"model_provenance":prov,"constructor":vars(args_for(arch)),"transform_policy":"fit-only fixed 51 positions; no pruning","batch_policy":policy,"selection":"minimum QPU inner-validation raw-seconds MSE; earliest epoch tie"})
    write_json(d/"transform.json",transform)
    opt=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4);best=float("inf");best_epoch=None;history=[]
    for epoch in range(1,501):
        model.train(); order=list(rows["fit"]);random.Random(seed*1000003+epoch).shuffle(order)
        if arch=="MLP":
            packs=[order[i:i+256] for i in range(0,len(order),256)]
        else:
            packs=[];cur=[];nodes=edges=0
            for r in order:
                a,b=sizes[r["canonical_observation_id"]]
                if cur and(nodes+a>GRAPH_NODE_CAP or edges+b>GRAPH_EDGE_CAP):packs.append(cur);cur=[];nodes=edges=0
                cur.append(r);nodes+=a;edges+=b
            if cur:packs.append(cur)
        for pack in packs:
            p,y=predict(pack);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(p,y);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            if arch=="MLP":
                p,y=predict(rows["validation"]);val=float(torch.nn.functional.mse_loss(p,y).item())
            else:
                squared=[]
                for pack in graph_packs(rows["validation"]):
                    p,y=predict(pack);squared.extend(((p-y)**2).detach().cpu().tolist())
                val=float(sum(squared)/len(squared))
        history.append({"epoch":epoch,"validation_mse":val})
        if val<best:
            best,best_epoch=val,epoch;payload={"model_state":model.state_dict(),"treatment":treatment,"architecture":arch,"initialization":spec["initialization"],"fold":fold,"seed":seed,"source_checkpoint_sha256":sha_file(source) if source else None,"transform_sha256":sha_file(d/"transform.json")};tmp=d/"best.pt.partial";tmp.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,tmp);os.replace(tmp,d/"best.pt")
        if epoch%10==0 or epoch==500:
            tmp=d/"current.pt.partial";torch.save({"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"epoch":epoch},tmp);os.replace(tmp,d/"current.pt")
    write_json(d/"training_history.json",history);selected=torch.load(d/"best.pt",map_location=device,weights_only=False);model.load_state_dict(selected["model_state"],strict=True);model.eval()
    predictions=[]
    with torch.no_grad():
        test_packs=[rows["test"]] if arch=="MLP" else graph_packs(rows["test"])
        for pack in test_packs:
            p,_=predict(pack)
            for r,value in zip(pack,p.detach().cpu().tolist()):
                predictions.append({"canonical_observation_id":r["canonical_observation_id"],"source_id":r["source_id"],"group_id":r["group_id"],"outer_fold":fold,"treatment":treatment,"architecture":arch,"initialization":spec["initialization"],"seed":seed,"target_seconds":r["target_seconds"],"prediction_seconds":float(value),"status":"PREDICTED","reason":"","checkpoint_sha256":sha_file(d/"best.pt"),"transform_sha256":sha_file(d/"transform.json")})
    fields=list(predictions[0]);write_csv(d/"predictions.csv",predictions,fields);term={"study_id":"postfreeze_track_a_transfer_v1","status":"COMPLETE","treatment":treatment,"fold":fold,"seed":seed,"best_epoch":best_epoch,"best_validation_mse":best,"checkpoint_sha256":sha_file(d/"best.pt"),"prediction_count":len(predictions),"source_checkpoint_sha256":sha_file(source) if source else None,"parameter_count":before["parameters"]};write_json(terminal,term);return term
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--treatment",choices=tuple(TREATMENTS),required=True);p.add_argument("--fold",type=int,choices=range(5),required=True);p.add_argument("--seed",type=int,choices=SEEDS,required=True);a=p.parse_args();print(json.dumps(fit_cell(a.treatment,a.fold,a.seed),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
