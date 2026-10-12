import csv
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with p.open() as stream:return list(csv.DictReader(stream))
class GridPublicationTests(unittest.TestCase):
 def test_exact_payload_hashes(self):
  m=json.loads((ROOT/'provenance/complete_grid_publication.json').read_text())
  self.assertFalse(m['private_row_level_QPU_included'])
  for r in m['files']:self.assertEqual(hashlib.sha256((ROOT/r['public_path']).read_bytes()).hexdigest(),r['published_sha256'])
 def test_complete_cases_and_seeds(self):
  base=ROOT/'results/generalization/full_grid'
  self.assertEqual(len(rows(base/'case_status.csv')),21)
  seeds=rows(base/'seed_metrics.csv');self.assertEqual(len(seeds),126)
  self.assertEqual(len({(r['case_id'],r['method_id'],r['seed']) for r in seeds}),126)
  self.assertEqual({int(r['seed']) for r in seeds},{42,1234,31415})
 def test_median_results_match_thesis(self):
  base=ROOT/'results/generalization/full_grid'
  ensemble={(r['case_id'],r['method_id']):r for r in rows(base/'median_metrics.csv') if r['scope']=='all_test'}
  self.assertEqual(len(ensemble),42)
  for r in rows(ROOT/'results/thesis/tables/c_qpu_generalization.csv'):
   original=ensemble[(r['case_id'],r['method_id'])]
   self.assertAlmostEqual(float(r['mae_seconds']),float(original['mae_seconds']),places=10)
   self.assertEqual(int(r['n']),int(original['predicted_rows']))
if __name__=='__main__':unittest.main()
