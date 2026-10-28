#!/usr/bin/env python3
"""Independent saved-input/checkpoint audit; no fitting or training."""
import collections
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmark_v1/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qualified_logical_benchmark as frozen
import run_mali_full_features as base
from shots_core import append_shots, expand_global_projection
from run import ShotsCache

OUT = ROOT / 'artifacts/postfreeze_mali_shots_ablation_v1/audit_preprocessing_v1'
ART = OUT.parent


def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    ctx = frozen.load_context()
    panel = {r['canonical_observation_id']: r for r in ctx['panel']}
    ctx['panel_by_id'] = panel
    canonical = {r['canonical_row_id']: r for r in rows(ROOT / 'artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv')}
    native = {r['canonical_row_id']: r for r in rows(ROOT / 'artifacts/real_qpu/qonductor_native_features/feature_rows.csv')}
    protocol = json.loads((ART / 'protocol.json').read_text())
    checks = []

    def check(name, result, detail=None):
        if not bool(result):
            raise RuntimeError(f'{name}: {detail}')
        checks.append({'check': name, 'status': 'PASS', 'detail': detail})

    check('baseline_predictions_match_locked_digest', sha(ART / 'baseline_predictions.csv') == protocol['baseline_predictions_sha256'])
    for relative, expected in protocol['source_files'].items():
        check('locked_input_or_source_' + relative, sha(ROOT / relative) == expected)
    reuse = json.loads((ART / 'baseline_reuse_receipt.json').read_text())
    for r in reuse['baseline_cell_checks']:
        mp = frozen.OUT / f"neural/fold_{r['fold']}/{r['method']}/seed_{r['seed']}/manifest.json"
        check('baseline_manifest_still_pinned_' + str(mp.relative_to(ROOT)), sha(mp) == r['manifest_sha256'])
    source_shots = collections.defaultdict(set)
    for cid, r in panel.items():
        s = float(r['shots'])
        raw = json.loads(canonical[cid]['raw_row_json'])
        check('shots_authority_' + cid, s > 0 and s.is_integer() and s == float(canonical[cid]['shots']) == float(native[cid]['shots']) == float(raw['shots']))
        source_shots[r['source_id']].add(int(s))
    fold_records, replay_records = [], []
    model_class, model_sha = base.load_pinned_model()
    for fold in range(5):
        parts = frozen.partitions(ctx, fold)
        tfm = json.loads((ART / f'training/fold_{fold}/transform.json').read_text())
        old = json.loads((frozen.OUT / f'neural/fold_{fold}/transform.json').read_text())
        check(f'fold{fold}_old_global_transform_exact', tfm['global'] == old['global'])
        check(f'fold{fold}_old_node_transform_exact', tfm['node'] == old['node'])
        check(f'fold{fold}_shot_fit_identity', tfm['fit_ids_sha256'] == frozen.idhash(parts['fit']))
        for method in ['mali_global_mlp', 'mali_logical_graph']:
            for seed in [42,1234,31415]:
                bm=json.loads((frozen.OUT/f'neural/fold_{fold}/{method}/seed_{seed}/manifest.json').read_text())
                check(f'fold{fold}_{method}_{seed}_baseline_transform_pinned', bm['identity']['transform'] == base.stable_hash(old))
        sets = {k: {panel[i]['group_id'] for i in parts[k]} for k in ['fit', 'validation', 'test']}
        check(f'fold{fold}_all_groups_disjoint', all(not sets[a] & sets[b] for a,b in [('fit','validation'),('fit','test'),('validation','test')]))
        x = np.asarray([float(panel[i]['shots']) for i in parts['fit']], dtype=np.float64)
        mean, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        check(f'fold{fold}_shot_stats_independent_fit_only', mean == tfm['shots']['mean'] and sd == tfm['shots']['std_ddof1'] and len(x) == tfm['shots']['fit_count'])
        # Independently rebuild the original retained columns and moments.
        a = np.asarray([[float(panel[i][f'g{j:02d}']) for j in range(51)] for i in parts['fit']], dtype=np.float64)
        retained = np.flatnonzero(a.sum(axis=0) > 0)
        check(f'fold{fold}_original_global_selection', retained.tolist() == tfm['global']['retained_indices'])
        check(f'fold{fold}_original_global_moments', np.array_equal(a[:, retained].mean(axis=0), tfm['global']['mean']) and np.array_equal(a[:, retained].std(axis=0, ddof=1), tfm['global']['std']))
        cache = ShotsCache(ctx, tfm, 32 * 1024**2)
        zs = {}
        for cid, r in panel.items():
            example = cache.get(cid, 'mali_full_features_mlp')
            expected = np.float32((float(r['shots']) - mean) / sd)
            original = base.transform_global([float(r[f'g{j:02d}']) for j in range(51)], old['global'])
            check(f'fold{fold}_actual_global_input_{cid}', np.array_equal(example.global_features[0, :-1].numpy(), original) and example.global_features[0,-1].item() == float(expected))
            zs[cid] = float(expected)
        fold_records.append({'fold': fold, 'fit_rows': len(parts['fit']), 'validation_rows': len(parts['validation']), 'test_rows': len(parts['test']), 'old_active_width': len(retained), 'new_active_width': len(retained)+1, 'shots_mean_fit':mean, 'shots_sample_std_fit':sd, 'fit_z_mean':float(np.mean([zs[i] for i in parts['fit']])), 'fit_z_std_ddof1':float(np.std([zs[i] for i in parts['fit']],ddof=1)), 'test_z_min':min(zs[i] for i in parts['test']), 'test_z_max':max(zs[i] for i in parts['test']), 'test_max_shots':max(float(panel[i]['shots']) for i in parts['test']), 'fit_max_shots':float(x.max())})
        # Repeated logical circuit/backend observations must keep distinct shots.
        by = collections.defaultdict(list)
        for cid in parts['test']:
            r = panel[cid]
            by[(r['source_qasm_sha256'], r['backend'])].append(cid)
        pairs = [v for v in by.values() if len({panel[i]['shots'] for i in v}) > 1]
        for v in pairs:
            u,vv = min(v,key=lambda i:float(panel[i]['shots'])), max(v,key=lambda i:float(panel[i]['shots']))
            check(f'fold{fold}_observation_cache_keeps_shots_{u}', cache.get(u,'mali_full_features_mlp').global_features[0,-1].item() != cache.get(vv,'mali_full_features_mlp').global_features[0,-1].item())
        fixtures = [i for i in parts['test'] if panel[i]['source_id']=='qonductor_single_circuit_ibm'] or parts['test']
        cid = min(fixtures, key=lambda i:int(panel[i]['node_count']))
        for tag, method, internal in [('M1','mali_global_mlp_shots','mali_full_features_mlp'),('G1','mali_logical_graph_shots','mali_full_features_graph')]:
            ex0 = base._make_example(cid,internal,ctx['ledger'],None,old,ctx['feature_root'],'cpu',False,set())
            ex1 = append_shots(ex0,zs[cid])
            check(f'fold{fold}_{tag}_global_prefix_unchanged',torch.equal(ex0.global_features,ex1.global_features[:,:-1]))
            if tag=='G1':
                check(f'fold{fold}_node_features_and_edges_unchanged',torch.equal(ex0.x,ex1.x) and torch.equal(ex0.edge_index,ex1.edge_index))
            for seed in [42,1234,31415]:
                d=ART/f'training/fold_{fold}/{method}/seed_{seed}'
                ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
                expected_tfm_hash=hashlib.sha256(json.dumps(tfm,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                check(f'trained_transform_identity_{fold}_{tag}_{seed}',ck['identity']['transform_sha256']==expected_tfm_hash)
                m=base.make_model(model_class,internal,len(retained),checkpoint_layers=False)
                before={k:v.clone() for k,v in m.state_dict().items()}
                info=expand_global_projection(m,torch)
                check(f'init_expansion_{fold}_{tag}_{seed}',info['added_trainable_parameters']==64 and all(torch.equal(before[k],v) if k!='gf_linear1.weight' else torch.equal(before[k],v[:,:-1]) for k,v in m.state_dict().items()))
                m.load_state_dict(ck['best_model_state'],strict=True);m.eval()
                weights=m.gf_linear1.weight[:,-1].detach()
                check(f'learned_shot_weights_{fold}_{tag}_{seed}',bool(torch.isfinite(weights).all()) and float(weights.abs().max())>0)
                param_id=ck['optimizer_state']['param_groups'][0]['params'][0]
                moment=ck['optimizer_state']['state'][param_id]['exp_avg_sq'][:,-1]
                check(f'optimizer_updated_shot_column_{fold}_{tag}_{seed}',float(moment.max())>0)
                batch=base._batch_to_device([ex1],'cpu',tag=='G1')
                tick=time.perf_counter()
                with torch.no_grad():actual=float(m(batch).reshape(-1)[0])
                infer_seconds=time.perf_counter()-tick
                saved=next(float(r['predicted_seconds']) for r in rows(d/'predictions.csv') if r['canonical_observation_id']==cid)
                check(f'saved_prediction_cpu_replay_{fold}_{tag}_{seed}',math.isclose(actual,saved,rel_tol=1e-4,abs_tol=1e-4),{'saved':saved,'cpu_replay':actual})
                cf=[]
                for shots in [1000,4000,20000]:
                    z=float(np.float32((shots-mean)/sd))
                    be=base._batch_to_device([append_shots(ex0,z)],'cpu',tag=='G1')
                    with torch.no_grad():cf.append(float(m(be).reshape(-1)[0]))
                replay_records.append({'fold':fold,'treatment':tag,'seed':seed,'fixture_id':cid,'nodes':int(panel[cid]['node_count']),'observed_shots':int(float(panel[cid]['shots'])),'saved_prediction':saved,'cpu_replay':actual,'abs_replay_difference':abs(actual-saved),'learned_shot_weight_max_abs':float(weights.abs().max()),'optimizer_shot_second_moment_max':float(moment.max()),'same_circuit_prediction_1000':cf[0],'same_circuit_prediction_4000':cf[1],'same_circuit_prediction_20000':cf[2],'cpu_replay_seconds':infer_seconds})
                del ck,m
        print('fold',fold,'PASS',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('fold_preprocessing.csv',fold_records),('checkpoint_input_replay.csv',replay_records)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    receipt={'status':'PASS','scope':'independent preprocessing, actual input, learned weights and saved checkpoint CPU replay','checks_count':len(checks),'folds':5,'global_input_rows_checked':4515*5,'checkpoint_replays':30,'source_shots':{k:sorted(v) for k,v in source_shots.items()},'model_source_sha256':model_sha,'max_cpu_prediction_replay_difference':max(r['abs_replay_difference'] for r in replay_records),'sensitive_fixture_checkpoints':sum(abs(r['same_circuit_prediction_20000']-r['same_circuit_prediction_1000'])>1e-6 for r in replay_records),'wall_seconds':time.perf_counter()-start,'cpu_threads':2,'no_training':True,'no_simulation':True,'checks':checks}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
