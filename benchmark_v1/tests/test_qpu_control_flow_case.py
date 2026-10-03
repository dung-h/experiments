import pytest

from benchmark_v1.scripts.run_qpu_control_flow_case import condition_values, node_rows, validate_partition, validate_layout_parity


def test_case_split_never_requires_test_label():
    canonical = {"train": {"target_seconds": "2"}, "case": {"target_seconds": "not_accessible"}}
    outer = {"train": {"outer_fold": "1", "unified_leakage_group_id": "a"},
             "case": {"outer_fold": "0", "unified_leakage_group_id": "b"}}
    features = {key: {"gate_counts_json": "{}"} for key in canonical}
    panel = {key: {"status": "eligible"} for key in canonical}
    assert validate_partition(canonical, outer, features, panel, "case", 2) == (["train"], ["case"])
    outer["case"]["unified_leakage_group_id"] = "a"
    with pytest.raises(ValueError, match="leaks groups"):
        validate_partition(canonical, outer, features, panel, "case", 2)


def test_new_dynamic_train_support_requires_review():
    canonical = {"train": {"target_seconds": "2"}, "case": {}}
    outer = {"train": {"outer_fold": "1", "unified_leakage_group_id": "a"},
             "case": {"outer_fold": "0", "unified_leakage_group_id": "b"}}
    features = {"train": {"gate_counts_json": '{"if_else":1}'}, "case": {}}
    panel = {key: {"status": "eligible"} for key in canonical}
    with pytest.raises(ValueError, match="support changed"):
        validate_partition(canonical, outer, features, panel, "case", 2)


def test_node_layout_conditions_and_target_rejection():
    rec = dict(names=["x", "if_else_control"], kinds=["operation", "control"], arity=[1, 1],
               num_qubits=2, num_clbits=2, parameter_presence=[0, 0],
               q0=[0, 1], q1=[-1, -1], q2=[-1, -1], q3=[-1, -1],
               branch_records=[{"control_node": 1, "condition_structure": {
                   "predicate_present": True, "comparison_value": True,
                   "classical_bit_index": 1, "register_size": 2}}])
    rows = node_rows(rec, {"x": 0, "__UNK_GATE__": 1})
    assert len(rows[0]) == 20
    assert rows[0][-4:] == [0, 0, 0, 0]
    assert rows[1][-4:] == condition_values(rec["branch_records"][0]["condition_structure"], 2)
    assert rows[1][1] == 1  # Structural kind does not expand train gate vocabulary.
    rec["target_seconds"] = 0
    with pytest.raises(ValueError, match="forbidden"):
        node_rows(rec, {"x": 0, "__UNK_GATE__": 1})


def test_signed_layout_mismatch_blocks_lift():
    from types import SimpleNamespace
    record = dict(names=["x"], arity=[1], num_qubits=2, parameter=[0],
                  q0=[0], q1=[-1], q2=[-1], q3=[-1])
    expected = dict(gate_one_hot=[1, 0], arity_one_hot=[0, 1, 0, 0],
                    parameter_presence=0, kind_one_hot=[1, 0, 0, 0, 0],
                    condition_channels=[0, 0, 0, 0])
    adapter = SimpleNamespace(encode_v4_node_layout=lambda rec, vocab: {"rows": [expected]})
    validate_layout_parity(record, {"x": 0, "__UNK_GATE__": 1}, adapter)
    expected["kind_one_hot"] = [0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="signed S79"):
        validate_layout_parity(record, {"x": 0, "__UNK_GATE__": 1}, adapter)


def test_signed_position_layout_is_independently_checked(monkeypatch):
    from types import SimpleNamespace
    from benchmark_v1.scripts import run_qpu_control_flow_case as runner
    record = dict(names=["cx"], arity=[2], num_qubits=5, parameter=[0],
                  q0=[1], q1=[3], q2=[-1], q3=[-1])
    vocab = {"cx": 0, "__UNK_GATE__": 1}
    expected = dict(gate_one_hot=[1, 0], arity_one_hot=[0, 0, 1, 0],
                    parameter_presence=0, kind_one_hot=[1, 0, 0, 0, 0],
                    condition_channels=[0, 0, 0, 0])
    adapter = SimpleNamespace(encode_v4_node_layout=lambda rec, vocab: {"rows": [expected]})
    assert runner.node_rows(record, vocab)[0][7:11] == [.25, .5, .75, 1.0]
    runner.validate_layout_parity(record, vocab, adapter)
    original = runner.node_rows
    def faulty_rows(rec, voc):
        rows = original(rec, voc)
        rows[0][8] = 0.0
        return rows
    monkeypatch.setattr(runner, "node_rows", faulty_rows)
    with pytest.raises(ValueError, match="signed S79"):
        runner.validate_layout_parity(record, vocab, adapter)
