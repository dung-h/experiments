#!/usr/bin/env python3
"""Post-hoc descriptive error audit of saved OOF predictions; no model fitting."""
import collections,csv,hashlib,json,time,sqlite3,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/postfreeze_mali_shots_ablation_v1'
OUT=ART/'failure_analysis_v1'

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def write(name,data):
    if not data:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    panel={r['canonical_observation_id']:r for r in read(ROOT/'artifacts/real_qpu/qualified_logical_panel/panel.csv')}
    canonical={r['canonical_row_id']:r for r in read(ROOT/'artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv')}
    outer={r['canonical_observation_id']:r for r in read(ROOT/'artifacts/real_qpu/qualified_logical_panel/outer_splits.csv')}
    inner=read(ROOT/'artifacts/real_qpu/qualified_logical_panel/inner_splits.csv')
    archive=ROOT.parent/'Qonductor-SC25/data/database/circuits.zip'
    db=ROOT.parent/'Qonductor-SC25/data/database/quantum_scheduler.db'
    provenance=[]
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as con,zipfile.ZipFile(archive) as z:
        for cid,r in panel.items():
            if r['source_id']!='qonductor_single_circuit_ibm':continue
            raw=json.loads(canonical[cid]['raw_row_json']);jid=int(raw['job_id'])
            job=con.execute('select ibm_quantum_id,backend_name,shots,taken_time,date from job where id=?',(jid,)).fetchone()
            files=[x[0] for x in con.execute('select filename from circuit where job_id=?',(jid,))]
            assert len(files)==1 and files[0]==canonical[cid]['qasm_path_or_member']
            physical=hashlib.sha256(z.read(files[0])).hexdigest()
            assert physical==r['compiled_qasm_sha256']==canonical[cid]['qasm_bytes_sha256']
            assert job[0]==raw['ibm_quantum_id'] and job[1]==r['backend'] and float(job[2])==float(r['shots']) and float(job[3])==float(r['target_seconds'])
            provenance.append({'observation_id':cid,'job_id':jid,'ibm_job_id':job[0],'backend':job[1],'shots':job[2],'database_taken_time':job[3],'date':job[4],'archive_member':files[0],'physical_sha256':physical,'all_checks':'PASS'})
    write('qonductor_raw_provenance.csv',provenance)
    preds={t:{} for t in ['M0','M1','G0','G1']}
    for r in read(ART/'aggregate/oof_predictions.csv'):
        cid=r['canonical_observation_id'];assert int(r['outer_fold'])==int(outer[cid]['outer_fold'])
        assert float(r['actual_seconds'])==float(panel[cid]['target_seconds'])
        preds[r['treatment']][cid]=float(r['predicted_seconds'])
    assert all(set(x)==set(panel) for x in preds.values())
    names=json.loads((ROOT/'benchmark_v1/protocol/mali_full_features.json').read_text())['global_features']['ordered_fields']
    support={};splits=[];features={}
    for fold in range(5):
        test=sorted(i for i in panel if int(outer[i]['outer_fold'])==fold)
        assignment={r['canonical_observation_id']:int(r['inner_fold']) for r in inner if int(r['outer_fold'])==fold}
        fit=sorted(i for i,k in assignment.items() if k!=0);valid=sorted(i for i,k in assignment.items() if k==0)
        tf=json.loads((ART/f'training/fold_{fold}/transform.json').read_text())
        base_tf=json.loads((ROOT/f'artifacts/real_qpu/qualified_logical_panel/neural/fold_{fold}/transform.json').read_text())
        assert tf['global']==base_tf['global'] and tf['node']==base_tf['node']
        for method in ['mali_logical_graph','mali_global_mlp']:
            for seed in [42,1234,31415]:
                m=json.loads((ROOT/f'artifacts/real_qpu/qualified_logical_panel/neural/fold_{fold}/{method}/seed_{seed}/manifest.json').read_text())
                assert m['identity']['test_ids']==hashlib.sha256('\n'.join(test).encode()).hexdigest()
        g={k:{panel[i]['group_id'] for i in ids} for k,ids in [('fit',fit),('validation',valid),('test',test)]}
        assert not g['fit']&g['validation'] and not g['fit']&g['test'] and not g['validation']&g['test']
        splits.append({'fold':fold,'fit_rows':len(fit),'fit_groups':len(g['fit']),'validation_rows':len(valid),'validation_groups':len(g['validation']),'outer_test_rows':len(test),'outer_test_groups':len(g['test']),'baseline_test_ids_identical':True,'baseline_transforms_identical':True})
        for name,ids in [('fit',fit),('validation',valid),('test',test)]:
            splits[-1]['qpack_'+name+'_logical_widths']=';'.join(map(str,sorted({int(float(panel[i]['g44'])) for i in ids if panel[i]['source_id']=='qpack_mcp'})))
        x=np.array([[float(panel[i][f'g{j:02d}']) for j in range(51)] for i in fit]);lo=x.min(0);hi=x.max(0)
        retained=np.array(tf['global']['retained_indices']);variable=np.array(tf['global']['variable_mask']);mean=np.array(tf['global']['mean'],dtype=np.float32);sd=np.array(tf['global']['std'],dtype=np.float32)
        dropped=[j for j in range(51) if j not in retained]
        for cid in test:
            r=panel[cid];raw=np.array([float(r[f'g{j:02d}']) for j in range(51)],dtype=np.float32)
            z=np.zeros(len(retained),dtype=np.float32);z[variable]=(raw[retained][variable]-mean[variable])/sd[variable]
            shot_z=float(np.float32((float(r['shots'])-tf['shots']['mean'])/tf['shots']['std_ddof1']))
            k=int(np.argmax(np.abs(z)));support[cid]={'max_abs_global_z':float(np.abs(z[k])),'largest_z_feature':names[int(retained[k])],'global_features_outside_fit_range':int(((raw<lo)|(raw>hi)).sum()),'test_active_dropped_features':int(sum(raw[j]!=0 for j in dropped)),'shots_z':shot_z,'shots_outside_fit_range':not min(float(panel[i]['shots']) for i in fit)<=float(r['shots'])<=max(float(panel[i]['shots']) for i in fit)}
            features[cid]=z.tobytes()
    write('split_verification.csv',splits)
    paired=[];slices=[];groups=[];sources=[];worst=[];classes=[];corr=[]
    for arch,old,new in [('MLP','M0','M1'),('Graph+MLP','G0','G1')]:
        rr=[]
        for cid,p in panel.items():
            y=float(p['target_seconds']);a=preds[old][cid];b=preds[new][cid];ea=abs(a-y);eb=abs(b-y);c=canonical[cid]
            rr.append({'architecture':arch,'canonical_observation_id':cid,'source_id':p['source_id'],'backend':p['backend'],'family':c['family'] or 'UNKNOWN','group_id':p['group_id'],'outer_fold':int(outer[cid]['outer_fold']),'shots':int(float(p['shots'])),'logical_width':float(p['g44']),'logical_depth':float(p['g45']),'target_seconds':y,'baseline_prediction_seconds':a,'shots_prediction_seconds':b,'baseline_abs_error':ea,'shots_abs_error':eb,'delta_abs_error':eb-ea,'baseline_signed_error':a-y,'shots_signed_error':b-y,**support[cid],'qasm_path':c['qasm_path_or_member'] or 'UNKNOWN'})
        paired+=rr
        for field in ['source_id','backend','shots','outer_fold','family']:
            by=collections.defaultdict(list)
            for r in rr:by[str(r[field])].append(r)
            for value,items in by.items():
                y=np.array([r['target_seconds'] for r in items]);a=np.array([r['baseline_abs_error'] for r in items]);b=np.array([r['shots_abs_error'] for r in items])
                gm=collections.defaultdict(list)
                for r in items:gm[r['group_id']].append(r)
                ga=np.mean([np.mean([r['baseline_abs_error'] for r in z]) for z in gm.values()]);gb=np.mean([np.mean([r['shots_abs_error'] for r in z]) for z in gm.values()])
                slices.append({'architecture':arch,'slice_type':field,'slice_value':value,'rows':len(items),'groups':len(gm),'baseline_pooled_mae':float(a.mean()),'shots_pooled_mae':float(b.mean()),'delta_pooled_mae':float((b-a).mean()),'baseline_group_balanced_mae':float(ga),'shots_group_balanced_mae':float(gb),'delta_group_balanced_mae':float(gb-ga),'baseline_bias':float(np.mean([r['baseline_signed_error'] for r in items])),'shots_bias':float(np.mean([r['shots_signed_error'] for r in items])),'baseline_error_gt_1s':int((a>1).sum()),'shots_error_gt_1s':int((b>1).sum()),'baseline_error_gt_2s':int((a>2).sum()),'shots_error_gt_2s':int((b>2).sum()),'baseline_error_gt_5s':int((a>5).sum()),'shots_error_gt_5s':int((b>5).sum()),'target_min':float(y.min()),'target_median':float(np.median(y)),'target_max':float(y.max()),'target_std':float(y.std()),'new_better_rows':int((b<a).sum()),'new_worse_rows':int((b>a).sum())})
        by=collections.defaultdict(list)
        for r in rr:by[r['group_id']].append(r)
        for gid,items in by.items():
            a=np.mean([r['baseline_abs_error'] for r in items]);b=np.mean([r['shots_abs_error'] for r in items])
            groups.append({'architecture':arch,'group_id':gid,'source':items[0]['source_id'],'family':items[0]['family'],'fold':items[0]['outer_fold'],'rows':len(items),'baseline_mae':float(a),'shots_mae':float(b),'delta_mae':float(b-a),'contribution_to_total_GB_delta':float((b-a)/165)})
        for s in sorted({r['source_id'] for r in rr}):
            items=[r for r in rr if r['source_id']==s];gg=[g for g in groups if g['architecture']==arch and g['source']==s]
            sources.append({'architecture':arch,'source':s,'rows':len(items),'groups':len(gg),'pooled_delta_contribution':sum(r['delta_abs_error'] for r in items)/4515,'group_balanced_delta_contribution':sum(g['contribution_to_total_GB_delta'] for g in gg),'groups_improved':sum(g['delta_mae']<0 for g in gg),'groups_worsened':sum(g['delta_mae']>0 for g in gg)})
        for label,key,reverse in [('worst_baseline','baseline_abs_error',True),('worst_with_shots','shots_abs_error',True),('most_worsened','delta_abs_error',True),('most_improved','delta_abs_error',False)]:
            for rank,r in enumerate(sorted(rr,key=lambda r:r[key],reverse=reverse)[:20],1):worst.append({'case_set':label,'rank':rank,**r})
        for source in ['ALL']+sorted({r['source_id'] for r in rr}):
            items=rr if source=='ALL' else [r for r in rr if r['source_id']==source]
            for feature in ['max_abs_global_z','global_features_outside_fit_range','shots_z']:
                x=np.array([r[feature] for r in items]);
                for t,err in [('baseline','baseline_abs_error'),('shots','shots_abs_error')]:
                    y=np.array([r[err] for r in items]);rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
                    corr.append({'architecture':arch,'source':source,'feature':feature,'prediction':t,'rows':len(items),'spearman_rho':rho,'interpretation':'post-hoc descriptive association; repeated rows not independent; no causal claim'})
        # Same actual model inputs within a fold: describe target variation.
        for tag in [old,new]:
            eq=collections.defaultdict(list)
            for r in rr:
                cid=r['canonical_observation_id'];p=panel[cid];key=features[cid]
                if arch=='Graph+MLP':key+=p['graph_sha256'].encode()
                if tag==new:key+=np.float32(r['shots_z']).tobytes()
                eq[(r['outer_fold'],hashlib.sha256(key).hexdigest())].append(r)
            for (fold,k),items in eq.items():
                if len(items)<2:continue
                pp=[preds[tag][r['canonical_observation_id']] for r in items];yy=[r['target_seconds'] for r in items]
                assert max(pp)-min(pp)<1e-5,(tag,fold,k,max(pp)-min(pp))
                ss=sorted({r['source_id'] for r in items});bb=sorted({r['backend'] for r in items});gg=sorted({r['group_id'] for r in items})
                classes.append({'architecture':arch,'treatment':tag,'fold':fold,'input_digest':k,'source':ss[0] if len(ss)==1 else 'MIXED','source_count':len(ss),'sources':';'.join(ss),'backend':bb[0] if len(bb)==1 else 'MULTIPLE','backend_count':len(bb),'backends':';'.join(bb),'group_count':len(gg),'rows':len(items),'shots_levels':';'.join(map(str,sorted({r['shots'] for r in items}))),'target_min':min(yy),'target_max':max(yy),'target_range':max(yy)-min(yy),'target_std':float(np.std(yy)),'prediction_min':min(pp),'prediction_max':max(pp),'example_id':items[0]['canonical_observation_id'],'group_id':items[0]['group_id']})
    context_rows=[];monotonic=[];context_counts={}
    for mode,field in [('logical','source_qasm_sha256'),('exact_physical','compiled_qasm_sha256')]:
        by=collections.defaultdict(list)
        for cid,r in panel.items():
            if r['source_id']=='qonductor_single_circuit_ibm':by[(r[field],r['backend'])].append(cid)
        multi={k:v for k,v in by.items() if len({panel[i]['shots'] for i in v})>1}
        context_counts[mode]={'all_hash_backend_contexts':len(by),'multishot_contexts':len(multi),'multishot_rows':sum(map(len,multi.values()))}
        for (h,backend),ii in multi.items():
            levels=sorted({panel[i]['shots'] for i in ii},key=float);series=collections.defaultdict(list)
            for s in levels:
                ids=[i for i in ii if panel[i]['shots']==s];yy=[float(panel[i]['target_seconds']) for i in ids];pm={t:float(np.median([preds[t][i] for i in ids])) for t in preds}
                times=sorted(canonical[i]['timestamp_utc'] for i in ids)
                context_rows.append({'context_kind':mode,'hash':h,'backend':backend,'shots':int(float(s)),'rows':len(ids),'physical_hashes_at_level':len({panel[i]['compiled_qasm_sha256'] for i in ids}),'target_min':min(yy),'target_median':float(np.median(yy)),'target_max':max(yy),**{'prediction_'+t:v for t,v in pm.items()},'first_date':times[0],'last_date':times[-1]})
                series['target'].append(float(np.median(yy)))
                for t,v in pm.items():series[t].append(v)
            for t,v in series.items():monotonic.append({'context_kind':mode,'hash':h,'backend':backend,'quantity':t,'shot_levels':len(levels),'decreasing_adjacent_steps':sum(b<a-1e-5 for a,b in zip(v,v[1:])),'interpretation':'observational label medians or saved model response; not a causal shot intervention'})
    for fn,data in [('paired_rows.csv',paired),('slice_errors.csv',slices),('group_effects.csv',groups),('source_contributions.csv',sources),('worst_cases.csv',worst),('identical_input_target_variation.csv',classes),('support_error_correlations.csv',corr)]:write(fn,data)
    write('qonductor_multishot_contexts.csv',context_rows);write('shot_response_monotonicity.csv',monotonic)
    influences=[]
    for arch in ['MLP','Graph+MLP']:
        gg=[r for r in groups if r['architecture']==arch];total=sum(r['delta_mae'] for r in gg)
        for r in gg:influences.append({**r,'primary_delta_all_165_groups':total/165,'descriptive_leave_one_group_delta':(total-r['delta_mae'])/164,'interpretation':'influence diagnostic only; all observations retained in primary score'})
    write('group_influence.csv',influences)
    selected={(r['architecture'],r['canonical_observation_id']) for r in worst}
    new_seeds={(r['treatment'],int(r['seed']),r['canonical_observation_id']):float(r['predicted_seconds']) for r in read(ART/'aggregate/seed_predictions.csv')}
    old_seeds={};manifests={}
    for fold in range(5):
        for arch,method,tag,newmethod in [('MLP','mali_global_mlp','M1','mali_global_mlp_shots'),('Graph+MLP','mali_logical_graph','G1','mali_logical_graph_shots')]:
            for seed in [42,1234,31415]:
                d=ROOT/f'artifacts/real_qpu/qualified_logical_panel/neural/fold_{fold}/{method}/seed_{seed}'
                for r in read(d/'predictions.csv'):old_seeds[arch,seed,r['canonical_observation_id']]=float(r['predicted_seconds'])
                oldm=json.loads((d/'manifest.json').read_text());newm=json.loads((ART/f'training/fold_{fold}/{newmethod}/seed_{seed}/manifest.json').read_text())
                manifests[arch,fold,seed]=(oldm['best_epoch'],newm['best_epoch_zero_based'])
    cases=[]
    for arch,cid in sorted(selected):
        tag='M1' if arch=='MLP' else 'G1';fold=int(outer[cid]['outer_fold']);y=float(panel[cid]['target_seconds'])
        oldtag='M0' if arch=='MLP' else 'G0'
        assert abs(float(np.median([old_seeds[arch,s,cid] for s in [42,1234,31415]]))-preds[oldtag][cid])<1e-6
        assert abs(float(np.median([new_seeds[tag,s,cid] for s in [42,1234,31415]]))-preds[tag][cid])<1e-6
        for seed in [42,1234,31415]:
            a=old_seeds[arch,seed,cid];b=new_seeds[tag,seed,cid];oe,ne=manifests[arch,fold,seed]
            cases.append({'architecture':arch,'observation_id':cid,'fold':fold,'seed':seed,'target_seconds':y,'baseline_prediction':a,'shots_prediction':b,'delta_absolute_error':abs(b-y)-abs(a-y),'baseline_best_epoch_zero_based':oe,'shots_best_epoch_zero_based':ne})
    write('case_seed_predictions.csv',cases)
    receipt={'status':'PASS','scope':'saved OOF failure analysis, post-hoc descriptive','same_outer_and_inner_assignments_as_baseline':True,'same_baseline_transform':True,'paired_rows':len(paired),'folds':5,'new_training':False,'new_model_inference':False,'new_simulations':False,'error_thresholds_seconds':[1,2,5],'threshold_status':'descriptive diagnostics chosen after results; not confirmatory endpoints','wall_seconds':time.perf_counter()-start,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ART/'aggregate/oof_predictions.csv',ROOT/'artifacts/real_qpu/qualified_logical_panel/panel.csv',ROOT/'artifacts/real_qpu/qualified_logical_panel/outer_splits.csv',ROOT/'artifacts/real_qpu/qualified_logical_panel/inner_splits.csv']}}
    receipt['qonductor_database_archive_checks']=len(provenance);receipt['qonductor_context_counts']=context_counts
    receipt['qonductor_unique_exact_physical_hashes']=len({r['physical_sha256'] for r in provenance})
    receipt['selected_case_seed_median_replay']='PASS';receipt['selected_architecture_cases']=len(selected)
    receipt['external_input_hashes']={str(p):sha(p) for p in [archive,db]}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
