import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completed_science',ROOT/'scripts/check_completed_science.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class CompletedScienceTests(unittest.TestCase):
 def test_complete_public_payload(self):
  self.assertEqual(module.verify()['status'],'PASS')
 def test_source_code_is_archived_not_public_training_authority(self):
  text=(ROOT/'methods/research_support/README.md').read_text()
  self.assertIn('Private QPU labels',text)
  self.assertIn('not portable public refit commands',text)
 def test_qpu_raw_overhead_not_exported(self):
  self.assertFalse((ROOT/'results/accuracy_cost/qpu/paired_observation_results.csv').exists())
  self.assertFalse((ROOT/'results/accuracy_cost/qpu/raw_b2_timings.csv').exists())
 def test_no_redundant_exhaustive_registry(self):
  self.assertFalse((ROOT/'results/validation/exhaustive/final_metric_registry.csv').exists())
  self.assertFalse((ROOT/'results/validation/exhaustive/final_comparison_registry.csv').exists())
 def test_no_models_environment_or_qasm_in_new_payload(self):
  for item in json.loads((ROOT/'provenance/completed_science.json').read_text())['files']:
   self.assertNotIn(Path(item['public_path']).suffix,{'.pt','.pth','.pkl','.npz','.qasm','.sqlite','.db','.log','.pyc'})
 def test_scientific_verdict(self):
  self.assertEqual(json.loads((ROOT/'provenance/completed_science.json').read_text())['scientific_verdict'],
    'SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS')

if __name__=='__main__':unittest.main()
