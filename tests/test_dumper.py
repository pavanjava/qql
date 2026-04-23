"""Tests for the QQL collection dumper (src/qql/dumper.py)."""
from __future__ import annotations

import pytest
from rich.console import Console

from qql.dumper import (
    _DUMP_BATCH_SIZE,
    _is_hybrid,
    _serialize_dict,
    _serialize_value,
    dump_collection,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def null_console() -> Console:
    return Console(quiet=True)


def _make_record(mocker, payload: dict, point_id="rec-1"):
    """Create a mock Qdrant ScoredPoint / Record with the given payload."""
    rec = mocker.MagicMock()
    rec.payload = payload
    rec.id = point_id
    return rec


def _make_client(mocker, *, exists=True, hybrid=False, points=None, total=None):
    """Build a mock QdrantClient for dump tests.

    *points* is a list of payload dicts.  scroll() returns them all in one
    batch when len(points) <= _DUMP_BATCH_SIZE, else two batches.
    """
    points = points or []
    client = mocker.MagicMock()
    client.collection_exists.return_value = exists

    # get_collection — return hybrid or dense vector config
    if hybrid:
        client.get_collection.return_value.config.params.vectors = {"dense": object()}
    else:
        # non-dict → dense-only
        client.get_collection.return_value.config.params.vectors = mocker.MagicMock(
            spec=[]  # not a dict
        )

    # count
    cnt = mocker.MagicMock()
    cnt.count = total if total is not None else len(points)
    client.count.return_value = cnt

    # scroll — single-batch by default
    records = [_make_record(mocker, p, f"id-{i}") for i, p in enumerate(points, 1)]
    client.scroll.return_value = (records, None)

    return client


# ── _serialize_value ──────────────────────────────────────────────────────────


class TestSerializeValue:
    def test_string(self):
        assert _serialize_value("hello world") == "'hello world'"

    def test_string_escapes_single_quote(self):
        assert _serialize_value("it's") == r"'it\'s'"

    def test_string_escapes_backslash(self):
        assert _serialize_value("a\\b") == "'a\\\\b'"

    def test_int(self):
        assert _serialize_value(42) == "42"

    def test_negative_int(self):
        assert _serialize_value(-7) == "-7"

    def test_float(self):
        result = _serialize_value(3.14)
        assert "3.14" in result

    def test_bool_true(self):
        assert _serialize_value(True) == "true"

    def test_bool_false(self):
        assert _serialize_value(False) == "false"

    def test_none(self):
        assert _serialize_value(None) == "null"

    def test_list(self):
        assert _serialize_value([1, 2, 3]) == "[1, 2, 3]"

    def test_nested_list_of_strings(self):
        result = _serialize_value(["a", "b"])
        assert result == "['a', 'b']"

    def test_dict_produces_braces(self):
        result = _serialize_value({"key": "val"})
        assert "{" in result and "}" in result
        assert "'key'" in result
        assert "'val'" in result


# ── _is_hybrid ────────────────────────────────────────────────────────────────


class TestIsHybrid:
    def test_dict_vectors_is_hybrid(self, mocker):
        client = mocker.MagicMock()
        client.get_collection.return_value.config.params.vectors = {"dense": object()}
        assert _is_hybrid("col", client) is True

    def test_scalar_vectors_is_not_hybrid(self, mocker):
        client = mocker.MagicMock()
        client.get_collection.return_value.config.params.vectors = mocker.MagicMock(
            spec=[]
        )
        assert _is_hybrid("col", client) is False


# ── dump_collection ───────────────────────────────────────────────────────────


class TestDumpCollection:
    def test_creates_output_file(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, points=[{"text": "hello"}])
        dump_collection("col", out, client, null_console(), null_console())
        assert (tmp_path / "dump.qql").exists()

    def test_writes_create_statement_dense(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, points=[{"text": "hello"}])
        dump_collection("my_col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "CREATE COLLECTION my_col\n" in content
        assert "HYBRID" not in content.split("CREATE")[1].split("\n")[0]

    def test_writes_create_statement_hybrid(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, hybrid=True, points=[{"text": "hello"}])
        dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "CREATE COLLECTION col HYBRID" in content

    def test_hybrid_insert_bulk_has_using_hybrid(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, hybrid=True, points=[{"text": "hello"}])
        dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "] USING HYBRID" in content

    def test_dense_insert_bulk_has_no_using_clause(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, points=[{"text": "hello"}])
        dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "USING HYBRID" not in content

    def test_skips_points_without_text_field(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        points = [{"text": "ok"}, {"author": "no_text_here"}, {"text": "also ok"}]
        client = _make_client(mocker, points=points)
        written, skipped = dump_collection("col", out, client, null_console(), null_console())
        assert written == 2
        assert skipped == 1
        content = (tmp_path / "dump.qql").read_text()
        assert "no_text_here" not in content

    def test_returns_zero_when_collection_missing(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, exists=False)
        written, skipped = dump_collection("missing", out, client, null_console(), null_console())
        assert written == 0
        assert skipped == 0

    def test_payload_values_serialized_correctly(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        payload = {"text": "hello", "year": 2024, "active": True, "score": 0.9}
        client = _make_client(mocker, points=[payload])
        dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "'year': 2024" in content
        assert "'active': true" in content
        assert "'score':" in content

    def test_dump_preserves_point_id_in_insert_values(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, points=[{"text": "hello"}])
        dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "'id': 'id-1'" in content

    def test_batches_multiple_scroll_pages(self, tmp_path, mocker):
        """When scroll returns two pages, two INSERT BULK blocks should be written."""
        out = str(tmp_path / "dump.qql")
        client = mocker.MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors = mocker.MagicMock(spec=[])
        cnt = mocker.MagicMock()
        cnt.count = _DUMP_BATCH_SIZE + 1
        client.count.return_value = cnt

        batch1 = [_make_record(mocker, {"text": f"doc {i}"}, f"id-{i}") for i in range(_DUMP_BATCH_SIZE)]
        batch2 = [_make_record(mocker, {"text": "last doc"}, "id-last")]
        # First scroll call returns batch1 with a non-None offset; second returns batch2 + None
        client.scroll.side_effect = [
            (batch1, "some_offset"),
            (batch2, None),
        ]

        written, skipped = dump_collection("col", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert written == _DUMP_BATCH_SIZE + 1
        assert content.count("INSERT BULK") == 2

    def test_header_contains_collection_name(self, tmp_path, mocker):
        out = str(tmp_path / "dump.qql")
        client = _make_client(mocker, points=[{"text": "x"}])
        dump_collection("medical_records", out, client, null_console(), null_console())
        content = (tmp_path / "dump.qql").read_text()
        assert "medical_records" in content.split("QQL Dump")[1]

    def test_output_file_created_in_nested_directory(self, tmp_path, mocker):
        out = str(tmp_path / "sub" / "dir" / "dump.qql")
        client = _make_client(mocker, points=[{"text": "x"}])
        dump_collection("col", out, client, null_console(), null_console())
        assert (tmp_path / "sub" / "dir" / "dump.qql").exists()
