"""Tests for the QQL script runner (src/qql/script.py)."""
from __future__ import annotations

import pytest
from rich.console import Console

from qql.ast_nodes import CreateCollectionStmt, InsertBulkStmt
from qql.exceptions import QQLRuntimeError
from qql.executor import ExecutionResult
from qql.lexer import Lexer
from qql.script import run_script, split_statements, strip_comments


# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenize(text: str):
    return Lexer().tokenize(text)


def null_console() -> Console:
    """A Console that writes to /dev/null — suppresses output in tests."""
    return Console(quiet=True)


# ── strip_comments ────────────────────────────────────────────────────────────

class TestStripComments:
    def test_removes_full_line_comment(self):
        result = strip_comments("-- this is a comment\nCREATE COLLECTION x")
        assert "-- this" not in result
        assert "CREATE" in result

    def test_removes_inline_comment(self):
        result = strip_comments("CREATE COLLECTION x -- inline note")
        assert "-- inline" not in result
        assert "CREATE COLLECTION x" in result

    def test_preserves_non_comment_lines(self):
        text = "CREATE COLLECTION x\nSHOW COLLECTIONS"
        assert strip_comments(text) == text

    def test_empty_string(self):
        assert strip_comments("") == ""

    def test_only_comments(self):
        result = strip_comments("-- line 1\n-- line 2")
        assert "line" not in result

    def test_comment_at_start_of_line(self):
        result = strip_comments("   -- leading spaces then comment\nDROP COLLECTION x")
        assert "DROP" in result
        assert "leading" not in result


# ── split_statements ──────────────────────────────────────────────────────────

class TestSplitStatements:
    def test_single_statement(self):
        tokens = tokenize("CREATE COLLECTION x")
        chunks = split_statements(tokens)
        assert len(chunks) == 1

    def test_two_statements(self):
        tokens = tokenize("CREATE COLLECTION x\nSHOW COLLECTIONS")
        chunks = split_statements(tokens)
        assert len(chunks) == 2

    def test_three_statements(self):
        tokens = tokenize(
            "CREATE COLLECTION x\n"
            "INSERT INTO COLLECTION x VALUES {'text': 'hi'}\n"
            "SHOW COLLECTIONS"
        )
        chunks = split_statements(tokens)
        assert len(chunks) == 3

    def test_bulk_insert_not_split_inside_brackets(self):
        """INSERT keyword inside a VALUES [...] array must NOT start a new chunk."""
        tokens = tokenize(
            "INSERT BULK INTO COLLECTION x VALUES [\n"
            "  {'text': 'a'},\n"
            "  {'text': 'b'}\n"
            "]\n"
            "SHOW COLLECTIONS"
        )
        chunks = split_statements(tokens)
        # There should be exactly 2 chunks: INSERT BULK and SHOW COLLECTIONS
        assert len(chunks) == 2

    def test_empty_input(self):
        tokens = tokenize("")
        chunks = split_statements(tokens)
        assert chunks == []

    def test_first_chunk_starts_with_create(self):
        tokens = tokenize("CREATE COLLECTION x\nDROP COLLECTION x")
        chunks = split_statements(tokens)
        from qql.lexer import TokenKind
        assert chunks[0][0].kind == TokenKind.CREATE
        assert chunks[1][0].kind == TokenKind.DROP


# ── run_script ────────────────────────────────────────────────────────────────

class TestRunScript:
    @pytest.fixture
    def script_file(self, tmp_path):
        """Factory: write content to a temp .qql file and return its path."""
        def _make(content: str) -> str:
            p = tmp_path / "test.qql"
            p.write_text(content)
            return str(p)
        return _make

    @pytest.fixture
    def mock_executor(self, mocker):
        ex = mocker.MagicMock()
        ex.execute.return_value = ExecutionResult(success=True, message="ok")
        return ex

    def test_executes_all_statements(self, script_file, mock_executor):
        path = script_file(
            "CREATE COLLECTION x\n"
            "SHOW COLLECTIONS\n"
        )
        ok, fail = run_script(path, mock_executor, null_console(), null_console())
        assert mock_executor.execute.call_count == 2
        assert ok == 2
        assert fail == 0

    def test_continues_on_error_by_default(self, script_file, mock_executor):
        mock_executor.execute.side_effect = [
            ExecutionResult(success=True, message="ok"),
            QQLRuntimeError("boom"),
            ExecutionResult(success=True, message="ok"),
        ]
        path = script_file(
            "CREATE COLLECTION x\n"
            "DROP COLLECTION missing\n"
            "SHOW COLLECTIONS\n"
        )
        ok, fail = run_script(path, mock_executor, null_console(), null_console())
        assert ok == 2
        assert fail == 1
        assert mock_executor.execute.call_count == 3

    def test_stops_on_error_when_flag_set(self, script_file, mock_executor):
        mock_executor.execute.side_effect = [
            QQLRuntimeError("fail fast"),
            ExecutionResult(success=True, message="ok"),
        ]
        path = script_file(
            "CREATE COLLECTION x\n"
            "SHOW COLLECTIONS\n"
        )
        ok, fail = run_script(
            path, mock_executor, null_console(), null_console(), stop_on_error=True
        )
        assert fail == 1
        assert mock_executor.execute.call_count == 1  # stopped after first

    def test_empty_script_returns_zero_counts(self, script_file, mock_executor):
        path = script_file("-- only comments\n\n")
        ok, fail = run_script(path, mock_executor, null_console(), null_console())
        assert ok == 0
        assert fail == 0
        mock_executor.execute.assert_not_called()

    def test_comments_are_stripped(self, script_file, mock_executor):
        path = script_file(
            "-- header comment\n"
            "CREATE COLLECTION x  -- inline comment\n"
        )
        ok, fail = run_script(path, mock_executor, null_console(), null_console())
        assert ok == 1
        assert fail == 0

    def test_nonexistent_file_returns_failure(self, mock_executor):
        ok, fail = run_script(
            "/no/such/file.qql", mock_executor, null_console(), null_console()
        )
        assert ok == 0
        assert fail == 1
