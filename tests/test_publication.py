from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_publication", ROOT / "scripts/verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class PublicationTests(unittest.TestCase):
    def fixture(self, root: Path):
        (root / "provenance").mkdir()
        (root / "README.md").write_text("# Fixture\n")
        (root / "provenance/publication.json").write_text(json.dumps({
            "third_party_qpu_observations_included": False,
            "third_party_qpu_row_level_derivatives_included": False,
            "source_circuit_bytes_included": False,
        }))
        self.index(root)

    def index(self, root):
        rows = [{"path": path.relative_to(root).as_posix(),
                 "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in root.rglob("*") if path.is_file()
                and path.relative_to(root).as_posix() != "provenance/files.json"]
        (root / "provenance/files.json").write_text(json.dumps({"files": rows}))

    def test_current_checkout(self):
        self.assertGreater(VERIFY.verify(ROOT), 100)

    def test_pasqal_missing_dispatcher_not_advertised_ready(self):
        protocol = json.loads((ROOT / "protocol/reproduction.json").read_text())
        capability = next(row for family in protocol["families"]
                          for row in family["capabilities"]
                          if row["capability_id"] == "analog_program_preprocessing")
        self.assertEqual(capability["status"], "not_ready")
        self.assertIsNone(capability["entrypoint"])
        self.assertFalse((ROOT / "scripts/benchmark.py").exists())
        source_path = capability["python_function"].split(":", 1)[0]
        self.assertTrue((ROOT / source_path).is_file())

    def test_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "README.md").write_text("altered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                VERIFY.verify(root)

    def test_unindexed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "scratch.log").write_text("scratch")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                VERIFY.verify(root)

    def test_qpu_observations_rejected_even_if_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "data/real_qpu/labels.csv"
            path.parent.mkdir(parents=True)
            path.write_text("target_seconds\n1\n")
            self.index(root)
            with self.assertRaisesRegex(ValueError, "excluded input"):
                VERIFY.verify(root)

    def test_broken_link_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "README.md").write_text("[missing](missing.md)\n")
            self.index(root)
            with self.assertRaisesRegex(ValueError, "broken documentation link"):
                VERIFY.verify(root)


if __name__ == "__main__":
    unittest.main()
