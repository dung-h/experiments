"""
Provides compile functions and architecture graphs
for the compile script.
"""

import os
from datetime import datetime

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

from pytket.architecture import Architecture
from pytket.qasm import circuit_from_qasm
from pytket import OpType
from pytket.passes import AutoRebase
from pytket.passes import FullPeepholeOptimise
from pytket.passes import DefaultMappingPass
from pytket.passes import SynthesiseTket
from pytket.qasm import circuit_to_qasm

from bqskit.ir import Circuit as BQSKitCircuit
from bqskit.ext import bqskit_to_qiskit
from bqskit.ext import qiskit_to_bqskit


def comp_qiskit(architecture, device_name, num_reps, in_qasm_dir, out_qasm_dir):
    """
    Compiles from .qasm files using the current qiskit version
    with optimization level 3.

    Args:
        architecture (str): The name of the device architecture. It is an unused dummy
            variable used to keep arguments consistent with comp_tket.

        device_name (str): The name of the device to compile the circuits for. Used to
            obtain an IBM backend.

        num_reps (int): The number of times to compile each circuit. Only the result with
            the lowest traditional depth is kept.

        in_qasm_dir (str): The directory containing the QASM files of the circuits to be
            compiled.

        out_qasm_dir (str): The directory in which to output the QASM files of the
            compiled circuits.
    """

    backend = QiskitRuntimeService().backend(device_name)

    for file_name in os.listdir(in_qasm_dir):
        if file_name.endswith(".qasm"):
            now = datetime.now()
            print(f"Started compiling {file_name} at {now.strftime('%H:%M:%S')}")

            # Load circuit via bqskit
            load_path = os.path.join(in_qasm_dir, file_name)
            circ = BQSKitCircuit.from_file(f"{load_path}")
            q_circ = bqskit_to_qiskit(circ)

            # Transpile without optimization via qiskit
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
            best_circ = pm.run(q_circ)
            best_depth = best_circ.depth()

            # Repeat and keep lowest depth option
            for _ in range(num_reps - 1):
                current_circ = pm.run(q_circ)
                current_depth = current_circ.depth()
                if current_depth < best_depth:
                    best_circ = current_circ
                    best_depth = current_depth

            # Save to qasm
            bqskit_out_circuit = qiskit_to_bqskit(best_circ)
            base_file_name, _ = file_name.split(".")
            save_path = os.path.join(out_qasm_dir, f"{base_file_name}.qasm")
            bqskit_out_circuit.save(save_path)


def comp_tket(architecture, device_name, num_reps, in_qasm_dir, out_qasm_dir):
    """
    Compiles from .qasm files using the current TKET version.
    Args:
        architecture (str): The name of the device architecture. Used to select the
            correct coupling graph and gate set for the TKET passes.

        device_name (str): The name of the device to compile the circuits for. It is an
            unused dummy variable used to keep arguments consistent with comp_qiskit.

        num_reps (int): The number of times to compile each circuit. Only the result with
            the lowest traditional depth is kept.

        in_qasm_dir (str): The directory containing the QASM files of the circuits to be
            compiled.

        out_qasm_dir (str): The directory in which to output the QASM files of the
            compiled circuits.
    """

    # Build workflow using device characteristics
    if architecture == "eagle":
        gate_set = {
            OpType.ECR,
            OpType.Rz,
            OpType.SX,
            OpType.X,
        }
        arch = eagle_cg()
    elif architecture == "heron":
        gate_set = {
            OpType.CZ,
            OpType.Rz,
            OpType.SX,
            OpType.X,
        }
        arch = heron_cg()

    pass_list = [
        AutoRebase(gate_set),
        FullPeepholeOptimise(),
        DefaultMappingPass(Architecture(arch)),
        AutoRebase(gate_set),
        SynthesiseTket(),
        AutoRebase(gate_set),
    ]

    for file_name in os.listdir(in_qasm_dir):
        if file_name.endswith(".qasm"):
            now = datetime.now()
            print(f"Started compiling {file_name} at {now.strftime('%H:%M:%S')}")

            # Load circuit via tket
            load_path = os.path.join(in_qasm_dir, file_name)
            circ = circuit_from_qasm(load_path)

            # Compile circuit via tket
            best_circ = apply_tket_passes(circ, pass_list)
            best_depth = best_circ.depth()

            # Repeat and keep lowest depth option
            for _ in range(num_reps - 1):
                current_circ = apply_tket_passes(circ, pass_list)
                current_depth = current_circ.depth()
                if current_depth < best_depth:
                    best_circ = current_circ
                    best_depth = current_depth

            # Save to qasm
            base_file_name, _ = file_name.split(".")
            save_path = os.path.join(out_qasm_dir, f"{base_file_name}.qasm")
            circuit_to_qasm(best_circ, save_path)


def apply_tket_passes(circuit, pass_list):
    """
    Applies TKET passes to circuit.
    Args:
        circuit (Circuit): The circuit to apply the passes to.

        pass_list (list(Pass)): The list of TKET passes to apply.
    """

    for tket_pass in pass_list:
        tket_pass.apply(circuit)
    return circuit


def eagle_cg():
    """
    Creates IBM Eagle coupling graph.
    """

    g = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (0, 14),
        (14, 18),
        (4, 15),
        (15, 22),
        (8, 16),
        (16, 26),
        (12, 17),
        (17, 30),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),
        (20, 33),
        (33, 39),
        (24, 34),
        (34, 43),
        (28, 35),
        (35, 47),
        (32, 36),
        (36, 51),
        (37, 38),
        (38, 39),
        (39, 40),
        (40, 41),
        (41, 42),
        (42, 43),
        (43, 44),
        (44, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        (48, 49),
        (49, 50),
        (50, 51),
        (37, 52),
        (52, 56),
        (41, 53),
        (53, 60),
        (45, 54),
        (54, 64),
        (49, 55),
        (55, 68),
        (56, 57),
        (57, 58),
        (58, 59),
        (59, 60),
        (60, 61),
        (61, 62),
        (62, 63),
        (63, 64),
        (64, 65),
        (65, 66),
        (66, 67),
        (67, 68),
        (68, 69),
        (69, 70),
        (58, 71),
        (71, 77),
        (62, 72),
        (72, 81),
        (66, 73),
        (73, 85),
        (70, 74),
        (74, 89),
        (75, 76),
        (76, 77),
        (77, 78),
        (78, 79),
        (79, 80),
        (80, 81),
        (81, 82),
        (82, 83),
        (83, 84),
        (84, 85),
        (85, 86),
        (86, 87),
        (87, 88),
        (88, 89),
        (75, 90),
        (90, 94),
        (79, 91),
        (91, 98),
        (83, 92),
        (92, 102),
        (87, 93),
        (93, 106),
        (94, 95),
        (95, 96),
        (96, 97),
        (97, 98),
        (98, 99),
        (99, 100),
        (100, 101),
        (101, 102),
        (102, 103),
        (103, 104),
        (104, 105),
        (105, 106),
        (106, 107),
        (107, 108),
        (96, 109),
        (100, 110),
        (110, 118),
        (104, 111),
        (111, 122),
        (108, 112),
        (112, 126),
        (113, 114),
        (114, 115),
        (115, 116),
        (116, 117),
        (117, 118),
        (118, 119),
        (119, 120),
        (120, 121),
        (121, 122),
        (122, 123),
        (123, 124),
        (124, 125),
        (125, 126),
    ]  # ibm_washington coupling map from quantum-computing.ibm.com

    return g


def heron_cg():
    """
    Creates IBM Heron coupling graph.
    """

    g = []

    # Add edges in horizontal rows
    for row in range(8):
        for col in range(
            15
        ):  # Stop short because right-most node has no edges further right
            edge = [(row * 20) + col, (row * 20) + col + 1]
            reverse_edge = [edge[1], edge[0]]
            g += [edge, reverse_edge]

    # Add vertical connections between rows
    for row in range(7):  # Stop one short because bottom row has no edges below it
        if (row % 2) == 0:
            # Num represents number of vertical links, going left to right
            for num, col in zip(range(4), range(3, 16, 4)):
                edges = [
                    [(row * 20) + 16 + num, (row * 20) + col],
                    [(row * 20) + 16 + num, ((row + 1) * 20) + col],
                ]
                reverse_edges = []
                for edge in edges:
                    reverse_edge = [edge[1], edge[0]]
                    reverse_edges.append(reverse_edge)
                g += edges
                g += reverse_edges
        else:
            for num, col in zip(range(4), range(1, 14, 4)):
                edges = [
                    [(row * 20) + 16 + num, (row * 20) + col],
                    [(row * 20) + 16 + num, ((row + 1) * 20) + col],
                ]
                reverse_edges = []
                for edge in edges:
                    reverse_edge = [edge[1], edge[0]]
                    reverse_edges.append(reverse_edge)
                g += edges
                g += reverse_edges

    return g
