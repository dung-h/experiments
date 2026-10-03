import hashlib
import json
import shutil

import pytest

import benchmark_v1.scripts.resolve_maestro_optimizer_off_protocol as resolver


def _fixture_protocols(tmp_path, monkeypatch, patch_bytes=None):
    root = tmp_path / "candidate"
    parent = root / "benchmark_v1/protocol/maestro_cpu_component_benchmark.json"
    overlay = root / "benchmark_v1/protocol/maestro_cpu_component_optimizer_off.json"
    parent.parent.mkdir(parents=True)
    shutil.copyfile(resolver.DEFAULT_PARENT, parent)

    overlay_data = json.loads(resolver.DEFAULT_OVERLAY.read_text(encoding="utf-8"))
    if patch_bytes is not None:
        patch = root / "benchmark_v1/patches/test_fixture.patch"
        patch.parent.mkdir(parents=True)
        patch.write_bytes(patch_bytes)
        patch_hash = hashlib.sha256(patch_bytes).hexdigest()
        overlay_data["context_overrides"]["patch_path"] = str(patch.relative_to(root))
        overlay_data["context_overrides"]["patch_sha256"] = patch_hash
        monkeypatch.setattr(resolver, "EXPECTED_PATCH_SHA256", patch_hash)
    overlay.write_text(json.dumps(overlay_data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(resolver, "ROOT", root)
    return root, parent, overlay


def _native_fixtures(tmp_path):
    native = tmp_path / "maestro.so"
    boost = tmp_path / "libboost_json.so"
    openmp = tmp_path / "libgomp.so"
    native.write_bytes(b"fake-native-module")
    boost.write_bytes(b"fake-boost")
    openmp.write_bytes(b"fake-openmp")
    return native, boost, openmp


def test_resolved_optimizer_off_protocol_preserves_parent_panel_and_pins_runtime(tmp_path, monkeypatch):
    patch_bytes = b"synthetic unit-test patch fixture; not Maestro source"
    root, parent, overlay = _fixture_protocols(tmp_path, monkeypatch, patch_bytes)
    native, boost, openmp = _native_fixtures(tmp_path)

    resolved = resolver.resolve_protocol(parent, overlay, native, boost, openmp)

    assert resolved["protocol_id"] == "maestro-cpu-component-optimizer-off-v1"
    assert resolved["method_id"] == "maestro_style_cpu_sv_component_optimizer_off_adaptation"
    assert resolved["context"]["optimize_circuit"] is False
    assert resolved["measurement"]["clock"] == resolved["evaluation"]["evaluation_target_clock"]
    assert resolved["panel"] == json.loads(parent.read_text())["panel"]
    assert resolved["resolved_from"]["parent_protocol_sha256"] == resolver.sha256_file(parent)
    assert resolved["resolved_from"]["overlay_protocol_sha256"] == resolver.sha256_file(overlay)
    assert resolved["local_execution_environment"]["native_module_sha256"] == resolver.sha256_file(native)
    assert resolved["local_execution_environment"]["runtime_link_dependency"]["library_sha256"] == resolver.sha256_file(boost)
    assert resolved["local_execution_environment"]["openmp_runtime"]["library_sha256"] == resolver.sha256_file(openmp)
    assert resolved["local_execution_environment"]["optimizer_off_patch_sha256"] == hashlib.sha256(patch_bytes).hexdigest()
    assert resolver.ROOT == root


def test_missing_rights_held_patch_fails_closed_with_explanation(tmp_path, monkeypatch):
    _, parent, overlay = _fixture_protocols(tmp_path, monkeypatch)
    native, boost, openmp = _native_fixtures(tmp_path)

    with pytest.raises(FileNotFoundError, match="GPL-derived dependency withheld pending rights review"):
        resolver.resolve_protocol(parent, overlay, native, boost, openmp)
