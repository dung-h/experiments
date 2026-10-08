from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_dataset", ROOT / "scripts/verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class QpuDatasetTests(unittest.TestCase):
    def test_published_selection_and_split_counts(self):
        VERIFY.verify_qpu_dataset(ROOT)

    def test_qualified_reconstructions_are_not_exact_recovery(self):
        data = json.loads((ROOT / "protocol/real_qpu_dataset.json").read_text())
        self.assertFalse(data["selection"]["exact_historical_logical_instance_claim"])
        self.assertFalse(data["source_filters"]["qpack_mcp"]["per_iteration_angles_recovered"])
        self.assertEqual(data["sources"]["qpack_mcp"]["groups"], 6)
        self.assertEqual(data["source_filters"]["qonductor_single_circuit_ibm"]["retained"], 230)

    def test_qpu_paper_comparison_is_complete_and_has_five_columns(self):
        text = (ROOT / "docs/paper_comparison.md").read_text()
        names = ["Ma–Li graph transformer", "Qonductor polynomial",
                 "Scholten CLOPS model", "QCRE and Qiskit scheduled duration",
                 "Hyb-HANAS cost model"]
        for name in names:
            row = next(line for line in text.splitlines() if line.startswith("| " + name + " |"))
            self.assertEqual(len(row.split(" | ")), 5)
            self.assertNotIn("pending", row.split(" | ")[3].lower())
            self.assertIn("4,515", row.split(" | ")[3])

    def test_other_cohort_counts_rejected(self):
        data = copy.deepcopy(json.loads((ROOT / "protocol/real_qpu_dataset.json").read_text()))
        data["observations"] += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol").mkdir()
            (root / "protocol/real_qpu_dataset.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "selection counts"):
                VERIFY.verify_qpu_dataset(root)

    def test_old_scores_not_published_as_new_results(self):
        data = json.loads((ROOT / "protocol/real_qpu_dataset.json").read_text())
        self.assertFalse(data["status"]["scores_from_other_training_populations_reused"])
        self.assertEqual(data["status"]["complete_qpu_metrics"], "complete_independent_saved_evidence_qa_pass")
        self.assertTrue((ROOT / "results/real_qpu/method_comparison.csv").exists())
        for path in [ROOT / "README.md", ROOT / "AGENTS.md", *(ROOT / "docs").glob("*.md")]:
            text = path.read_text()
            for stale in ("8,767", "7,350", "8767", "7350"):
                self.assertNotIn(stale, text, str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
