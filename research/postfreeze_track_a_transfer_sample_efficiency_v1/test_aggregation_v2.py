"""Independent saved-output replay; no training or inference."""
import csv
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2'


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (OUT/'median_predictions.csv').open() as f: cls.rows=list(csv.DictReader(f))
        with (OUT/'paired_comparisons.csv').open() as f: cls.comps=list(csv.DictReader(f))

    def test_counts_and_coverage(self):
        self.assertEqual(len(self.rows),126420)
        sets=defaultdict(set)
        for r in self.rows:sets[(r['fraction'],r['subset_seed'],r['treatment'])].add(r['canonical_observation_id'])
        self.assertEqual(len(sets),28)
        reference=next(iter(sets.values()))
        self.assertEqual(len(reference),4515)
        self.assertTrue(all(s==reference for s in sets.values()))

    def test_independent_group_delta_and_bootstrap(self):
        archive=np.load(OUT/'bootstrap_distributions.npz',allow_pickle=False)
        rng=np.random.default_rng(42)
        fresh=rng.integers(0,165,size=(10000,165),dtype=np.uint16)
        np.testing.assert_array_equal(fresh,archive['draw_indices'])
        for c in self.comps:
            f=c['fraction'];left,right=c['contrast'].split(' - ')
            per=defaultdict(dict)
            for r in self.rows:
                if r['fraction']==f and r['treatment'] in (left,right):
                    per[(r['subset_seed'],r['canonical_observation_id'])][r['treatment']]=r
            grouped=defaultdict(list)
            for v in per.values():
                a,b=v[left],v[right]
                grouped[a['group_id']].append(abs(float(a['prediction_seconds'])-float(a['target_seconds']))-abs(float(b['prediction_seconds'])-float(b['target_seconds'])))
            # Equal observations per group across fixed draws: flattening their
            # errors equals averaging per-draw group errors, without pseudo-replication.
            v=np.array([sum(grouped[g])/len(grouped[g]) for g in archive['group_ids']])
            self.assertAlmostEqual(float(v.mean()),float(c['delta_group_mae']),places=10)
            dist=v[fresh].mean(1)
            np.testing.assert_allclose(dist,archive[f'{f}_{c["architecture"]}'],rtol=0,atol=1e-12)
            lo,hi=np.quantile(dist,[.025,.975])
            self.assertAlmostEqual(float(lo),float(c['ci95_low']),places=10)
            self.assertAlmostEqual(float(hi),float(c['ci95_high']),places=10)

    def test_receipt_inputs_unchanged(self):
        receipt=json.loads((OUT/'validation_receipt.json').read_text())
        self.assertEqual(receipt['status'],'PASS')
        for name,h in receipt['input_hashes'].items():
            if not name.endswith('best.pt'):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),h,name)


if __name__=='__main__':unittest.main()
