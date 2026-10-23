#!/usr/bin/env python3
"""Input/throughput diagnostic; discard all fitted parameters and predictions.

Wait for the current training cell to finish while its dispatcher is paused,
then compare original and doubled graph pack caps on one fixed 50% fit set.
The dispatcher always resumes, including on diagnostic failure.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatcher-pid', type=int, required=True)
    parser.add_argument('--label', default='batch_throughput_diagnostic')
    args = parser.parse_args()
    pid = args.dispatcher_pid
    # Validate the target before sending the stop signal.
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'run_safe_cells.py' not in cmd:
        raise RuntimeError('dispatcher_identity_mismatch')
    os.kill(pid, signal.SIGSTOP)
    try:
        child_file = Path(f'/proc/{pid}/task/{pid}/children')
        children = child_file.read_text().split()
        deadline = time.monotonic() + 950
        while children:
            states = []
            for child in children:
                status = Path(f'/proc/{child}/status')
                if status.exists():
                    state = next(s for s in status.read_text().splitlines() if s.startswith('State:'))
                    states.append(state.split()[1])
            if not states or all(s == 'Z' for s in states):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('training_cell_not_finished_no_benchmark_run')
            time.sleep(1)

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        import torch
        from torch_geometric.data import Batch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.95)
        script = Path(__file__).with_name('run_sample_efficiency.py')
        spec = importlib.util.spec_from_file_location('extension', script)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        base = extension.load_runner()
        panel = extension.read_csv(base.PANEL)
        by_id = {r['canonical_observation_id']: r for r in panel}
        parts, subset = extension.subset_plan(base, .5, 20261019, 0)
        rows = [by_id[i] for i in parts['fit']]
        random.Random(42 * 1000003 + 1).shuffle(rows)
        base._sample_prefetch_graphs(rows)
        gm, gs, _ = base.transforms(by_id, parts['fit'])
        nm, ns = base.graph_transforms(by_id, parts['fit'])
        data = {r['canonical_observation_id']: base.graph_item(r, gm, gs, nm, ns) for r in rows}
        results = []
        for factor in (1, 2):
            packs = []
            pack = []
            n = e = 0
            for row in rows:
                nn, ee = int(row['node_count']), int(row['edge_count'])
                if pack and (n + nn > factor * base.GRAPH_NODE_CAP or e + ee > factor * base.GRAPH_EDGE_CAP):
                    packs.append(pack)
                    pack = []
                    n = e = 0
                pack.append(row)
                n += nn
                e += ee
            if pack:
                packs.append(pack)
            base.seed_all(42)
            model, _ = base.model_for('GRAPH_MLP')
            model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            record = {'factor': factor, 'optimizer_steps_per_epoch': len(packs),
                      'max_nodes': max(sum(int(r['node_count']) for r in p) for p in packs),
                      'status': 'RUNNING'}
            try:
                torch.cuda.reset_peak_memory_stats()
                durations = []
                for iteration in range(4):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for p in packs:
                        batch = Batch.from_data_list([data[r['canonical_observation_id']] for r in p]).cuda()
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(batch).reshape(-1)
                        loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
                        loss.backward()
                        optimizer.step()
                        del batch, prediction, loss
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - start)
                record.update(status='PASS', warm_epoch_seconds=sum(durations[1:])/3,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            except torch.cuda.OutOfMemoryError as exc:
                record.update(status='CUDA_OOM', message=str(exc),
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            results.append(record)
            del optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
        result = {'diagnostic_only': True, 'subset': subset, 'results': results,
                  'predictions_saved': False, 'model_checkpoint_saved': False,
                  'existing_training_recipe_changed': False,
                  'allocator': os.environ['PYTORCH_CUDA_ALLOC_CONF']}
        dest = extension.OUT / f'{args.label}.json'
        dest.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
