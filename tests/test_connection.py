"""Tests for the Connection class (src/qql/connection.py).

All tests mock QdrantClient so no live Qdrant instance is required.
"""
import pytest

from qql import Connection, QQLConfig, Executor, ExecutionResult, run_query
from qql.exceptions import QQLSyntaxError


# ── TestConnectionInit ────────────────────────────────────────────────────────

class TestConnectionInit:
    """Connection.__init__ stores config and wires up the executor."""

    def test_default_url_and_no_secret(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        conn = Connection()
        assert conn.config.url == "http://localhost:6333"
        assert conn.config.secret is None

    def test_custom_url_and_secret_passed_to_qdrant_client(self, mocker):
        mock_client_cls = mocker.patch("qdrant_client.QdrantClient")
        Connection("https://cloud.example.io", secret="s3cr3t")
        mock_client_cls.assert_called_once_with(
            url="https://cloud.example.io", api_key="s3cr3t", verify=True
        )

    def test_ssl_verify_option_passed_to_qdrant_client(self, mocker):
        mock_client_cls = mocker.patch("qdrant_client.QdrantClient")
        conn = Connection("https://internal.example.io", verify=False)
        mock_client_cls.assert_called_once_with(
            url="https://internal.example.io", api_key=None, verify=False
        )
        assert conn.config.verify is False

    def test_custom_ca_bundle_passed_to_qdrant_client(self, mocker):
        mock_client_cls = mocker.patch("qdrant_client.QdrantClient")
        conn = Connection("https://internal.example.io", verify="/etc/ssl/internal-ca.pem")
        mock_client_cls.assert_called_once_with(
            url="https://internal.example.io",
            api_key=None,
            verify="/etc/ssl/internal-ca.pem",
        )
        assert conn.config.verify == "/etc/ssl/internal-ca.pem"

    def test_custom_default_model_stored_in_config(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        conn = Connection("http://localhost:6333", default_model="BAAI/bge-small-en-v1.5")
        assert conn.config.default_model == "BAAI/bge-small-en-v1.5"

    def test_config_and_executor_properties_return_correct_types(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        conn = Connection("http://localhost:6333")
        assert isinstance(conn.config, QQLConfig)
        assert isinstance(conn.executor, Executor)


# ── TestConnectionRunQuery ────────────────────────────────────────────────────

class TestConnectionRunQuery:
    """Connection.run_query() pipes through the Lexer → Parser → Executor."""

    def test_run_query_calls_executor_execute(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        mock_executor = mocker.MagicMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        mocker.patch("qql.connection.Executor", return_value=mock_executor)

        conn = Connection("http://localhost:6333")
        conn.run_query("SHOW COLLECTIONS")
        mock_executor.execute.assert_called_once()

    def test_executor_instance_reused_across_queries(self, mocker):
        """Executor() is constructed once; run_query() never re-instantiates it."""
        mocker.patch("qdrant_client.QdrantClient")
        mock_executor = mocker.MagicMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        executor_cls = mocker.patch("qql.connection.Executor", return_value=mock_executor)

        conn = Connection("http://localhost:6333")
        conn.run_query("SHOW COLLECTIONS")
        conn.run_query("SHOW COLLECTIONS")
        conn.run_query("SHOW COLLECTIONS")

        # Executor constructor called exactly once, not once per query
        executor_cls.assert_called_once()
        # But execute() called three times
        assert mock_executor.execute.call_count == 3

    def test_invalid_query_raises_qql_syntax_error(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        conn = Connection("http://localhost:6333")
        with pytest.raises(QQLSyntaxError):
            conn.run_query("TOTALLY INVALID QUERY GIBBERISH")

    def test_run_query_returns_execution_result(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        mock_executor = mocker.MagicMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="1 collection(s) found", data=["docs"]
        )
        mocker.patch("qql.connection.Executor", return_value=mock_executor)

        conn = Connection("http://localhost:6333")
        result = conn.run_query("SHOW COLLECTIONS")
        assert isinstance(result, ExecutionResult)
        assert result.success is True


# ── TestConnectionLifecycle ───────────────────────────────────────────────────

class TestConnectionLifecycle:
    """Connection.close() and the context-manager protocol."""

    def test_close_calls_client_close(self, mocker):
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        conn = Connection("http://localhost:6333")
        conn.close()
        mock_client.close.assert_called_once()

    def test_context_manager_enter_returns_self(self, mocker):
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        conn = Connection("http://localhost:6333")
        assert conn.__enter__() is conn

    def test_context_manager_exit_calls_close(self, mocker):
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        with Connection("http://localhost:6333"):
            pass
        mock_client.close.assert_called_once()

    def test_context_manager_closes_even_when_body_raises(self, mocker):
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        with pytest.raises(ValueError):
            with Connection("http://localhost:6333"):
                raise ValueError("simulated error inside with-block")
        # close() must still have been called
        mock_client.close.assert_called_once()


# ── TestRunQueryBackwardCompat ────────────────────────────────────────────────

class TestRunQueryBackwardCompat:
    """Standalone run_query() keeps working with the same signature and semantics."""

    def test_run_query_still_callable_with_same_signature(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        mock_executor = mocker.MagicMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        mocker.patch("qql.connection.Executor", return_value=mock_executor)
        # Must not raise; same kwargs as before the refactor
        run_query("SHOW COLLECTIONS", url="http://localhost:6333", secret=None)

    def test_run_query_delegates_to_connection(self, mocker):
        """run_query() must delegate to Connection, not re-implement the pipeline."""
        conn_instance = mocker.MagicMock()
        conn_instance.run_query.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        # Context-manager protocol: __enter__ returns the mock, __exit__ is a no-op
        conn_instance.__enter__ = mocker.MagicMock(return_value=conn_instance)
        conn_instance.__exit__ = mocker.MagicMock(return_value=False)
        conn_cls = mocker.patch("qql.Connection", return_value=conn_instance)
        run_query("SHOW COLLECTIONS", url="http://localhost:6333")
        conn_cls.assert_called_once_with(
            url="http://localhost:6333", secret=None, default_model=None, verify=True
        )
        conn_instance.run_query.assert_called_once_with("SHOW COLLECTIONS")

    def test_run_query_closes_connection_after_query(self, mocker):
        """run_query() must call close() — it must not leak the QdrantClient."""
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        mock_executor = mocker.MagicMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        mocker.patch("qql.connection.Executor", return_value=mock_executor)
        run_query("SHOW COLLECTIONS", url="http://localhost:6333")
        # close() must have been called exactly once
        mock_client.close.assert_called_once()

    def test_run_query_closes_connection_even_when_query_raises(self, mocker):
        """run_query() must call close() even if the query throws."""
        mock_client = mocker.MagicMock()
        mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
        # Make the query raise a runtime error (e.g. collection not found)
        from qql.exceptions import QQLRuntimeError
        mock_executor = mocker.MagicMock()
        mock_executor.execute.side_effect = QQLRuntimeError("collection 'x' does not exist")
        mocker.patch("qql.connection.Executor", return_value=mock_executor)
        with pytest.raises(QQLRuntimeError):
            run_query("SEARCH x SIMILAR TO 'q' LIMIT 5", url="http://localhost:6333")
        # close() still called
        mock_client.close.assert_called_once()

    def test_run_query_invalid_syntax_still_raises(self, mocker):
        mocker.patch("qdrant_client.QdrantClient")
        with pytest.raises(QQLSyntaxError):
            run_query("TOTALLY INVALID", url="http://localhost:6333")
