import csv
import io

import pytest

from scripts.materialize_csv_parts import materialize, split_csv


def test_roundtrip_preserves_multiline_crlf_and_no_final_newline(tmp_path):
    raw = b'a,b\r\n1,"two\r\nlines"\r\n2,"quoted ""text"""\r\n3,last'
    original = tmp_path / "input.csv"
    original.write_bytes(raw)
    manifest = split_csv(tmp_path, "input.csv", limit=34)
    assert len(manifest["parts"]) > 1
    assert manifest["rows"] == 3
    for part in manifest["parts"]:
        text = (tmp_path / part["path"]).read_bytes().decode()
        assert len(list(csv.reader(io.StringIO(text, newline="")))) - 1 == part["rows"]
    original.unlink()
    assert materialize(tmp_path, manifest) == "restored_and_verified"
    assert original.read_bytes() == raw
    assert materialize(tmp_path, manifest) == "already_verified"


def test_rejects_corrupt_part_without_leaving_output(tmp_path):
    original = tmp_path / "input.csv"
    original.write_bytes(b"a,b\n1,2\n3,4\n")
    manifest = split_csv(tmp_path, "input.csv", limit=9)
    original.unlink()
    part = tmp_path / manifest["parts"][0]["path"]
    part.write_bytes(part.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        materialize(tmp_path, manifest)
    assert not original.exists()
    assert not list(tmp_path.glob(".csv-join-*"))


def test_refuses_to_overwrite_existing_mismatched_data(tmp_path):
    original = tmp_path / "input.csv"
    original.write_bytes(b"a\n1\n")
    manifest = split_csv(tmp_path, "input.csv")
    original.write_bytes(b"user change\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        materialize(tmp_path, manifest)
    assert original.read_bytes() == b"user change\n"


def test_header_only_and_escaped_path(tmp_path):
    original = tmp_path / "input.csv"
    original.write_bytes(b"a,b\n")
    manifest = split_csv(tmp_path, "input.csv")
    original.unlink()
    materialize(tmp_path, manifest)
    assert original.read_bytes() == b"a,b\n"
    manifest["path"] = "../outside.csv"
    with pytest.raises(ValueError):
        materialize(tmp_path, manifest)
