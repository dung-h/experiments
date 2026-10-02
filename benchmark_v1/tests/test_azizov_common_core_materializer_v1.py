from __future__ import annotations

import json
import unittest
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter

from benchmark_v1.scripts import materialize_azizov_common_core_v1 as materializer


ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = json.loads((ROOT / "benchmark_v1/protocol/azizov_local_feature_dictionary_v1.json").read_text())
CAPS = {
    "max_operation_nodes": 300000,
    "max_edges": 1000000,
    "max_quantum_wires": 127,
    "max_classical_wires": 127,
    "max_real_parameters": 4,
}


class AzizovMaterializerTests(unittest.TestCase):
    def test_frozen_dimensions_and_order_are_exact(self) -> None:
        globals_ = DICTIONARY["ordered_global_fields"]
        extensions = DICTIONARY["ordered_compiled_extension_fields"]
        self.assertEqual(len(globals_), 51)
        self.assertEqual(len(set(globals_)), 51)
        self.assertEqual(len(extensions), 13)
        self.assertEqual(len(set(extensions)), 13)
        self.assertEqual(globals_[0], "count_u3")
        self.assertEqual(globals_[-1], "liveness")
        self.assertEqual(extensions[0], "compiled_depth")
        self.assertEqual(extensions[-1], "compiled_liveness")

    def test_graph_uses_quantum_and_classical_wire_dependencies(self) -> None:
        circuit = QuantumCircuit(3, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(0.5, 1)
        circuit.measure(1, 0)
        circuit.measure(2, 0)
        circuit.barrier(0, 2)

        record = materializer.graph_record(circuit, "a" * 64, "source", CAPS)
        self.assertEqual(record["node_count"], 6)
        self.assertEqual(record["representation"], "source")
        self.assertEqual(record["nodes"][2]["parameter_presence_masks"], [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(record["nodes"][2]["parameter_values_over_pi"][0], 0.5 / 3.141592653589793)
        self.assertEqual(record["nodes"][4]["cargs"], [0])
        self.assertEqual(record["nodes"][5]["qargs"], [0, 2])
        edges = {tuple(edge) for edge in record["edges"]}
        self.assertEqual(edges, {(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (4, 5)})
        self.assertEqual(record["edge_count"], len(edges))
        self.assertTrue(all(source < target for source, target in edges))

    def test_global_and_compiled_vectors_match_frozen_dimensions(self) -> None:
        circuit = QuantumCircuit(3, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(0.5, 1)
        circuit.measure(1, 0)
        circuit.measure(2, 0)
        circuit.barrier(0, 2)

        globals_ = materializer.global_features(circuit, DICTIONARY["ordered_global_fields"])
        extension = materializer.compiled_extension(circuit, DICTIONARY["ordered_compiled_extension_fields"])
        self.assertEqual(list(globals_), DICTIONARY["ordered_global_fields"])
        self.assertEqual(len(globals_), 51)
        self.assertEqual(list(extension), DICTIONARY["ordered_compiled_extension_fields"])
        self.assertEqual(len(extension), 13)
        self.assertEqual(globals_["count_cx"], 1.0)
        self.assertEqual(globals_["num_qubits"], 3.0)
        self.assertAlmostEqual(globals_["program_communication"], 1.0 / 3.0)
        self.assertAlmostEqual(globals_["entanglement_ratio"], 1.0 / 3.0)
        self.assertEqual(extension["compiled_measurement_count"], 2.0)

    def test_structural_degenerate_case_is_explicit_zero(self) -> None:
        metrics = materializer.structural_metrics(QuantumCircuit(1))
        self.assertEqual(metrics, {
            "program_communication": 0.0,
            "critical_depth": 0.0,
            "entanglement_ratio": 0.0,
            "parallelism": 0.0,
            "liveness": 0.0,
        })

    def test_nonreal_and_overwide_parameter_inputs_fail_closed(self) -> None:
        symbolic = QuantumCircuit(1)
        theta = Parameter("theta")
        symbolic.rz(theta, 0)
        with self.assertRaises(materializer.UnsupportedCircuit) as symbolic_error:
            materializer.graph_record(symbolic, "b" * 64, "source", CAPS)
        self.assertEqual(symbolic_error.exception.code, "unsupported_parameter")

        overwide = QuantumCircuit(1)
        overwide.append(Gate("five_parameter_gate", 1, [0.1, 0.2, 0.3, 0.4, 0.5]), [0])
        with self.assertRaises(materializer.UnsupportedCircuit) as width_error:
            materializer.graph_record(overwide, "c" * 64, "source", CAPS)
        self.assertEqual(width_error.exception.code, "parameter_count_cap")

    def test_c44_summary_extractor_counts_native_two_qubit_names(self) -> None:
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        summary = materializer.summary_metrics(circuit)
        self.assertEqual(summary["physical_width"], 2)
        self.assertEqual(summary["physical_ops"], 4)
        self.assertEqual(summary["physical_two_qubit_ops"], 1)
        self.assertEqual(summary["dag_node_count"], 4)

    def test_fold_audit_uses_frozen_big_endian_prefix(self) -> None:
        import hashlib

        digest = "0123456789abcdef" * 4
        expected = int.from_bytes(hashlib.sha256(("20260925-split|" + digest).encode()).digest()[:8], "big") % 5
        self.assertEqual(materializer.fold_for(digest), expected)


if __name__ == "__main__":
    unittest.main()
