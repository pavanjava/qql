"""Tests for the AsyncConnection and QQLAsyncBatch classes.

All tests mock AsyncQdrantClient so no live Qdrant instance is required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from qql import (
    AsyncConnection,
    QQLConfig,
    AsyncExecutor,
    ExecutionResult,
    QQLAsyncBatch,
)
from qql.exceptions import QQLSyntaxError


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ── TestAsyncConnectionInit ───────────────────────────────────────────────────

class TestAsyncConnectionInit:
    """AsyncConnection.__init__ stores config and wires up the async executor."""

    def test_default_url_and_no_secret(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        conn = AsyncConnection()
        assert conn.config.url == "http://localhost:6333"
        assert conn.config.secret is None

    def test_custom_url_and_secret_passed_to_async_qdrant_client(self, mocker):
        mock_client_cls = mocker.patch("qql.async_connection.AsyncQdrantClient")
        AsyncConnection("https://cloud.example.io", secret="s3cr3t")
        mock_client_cls.assert_called_once_with(
            url="https://cloud.example.io", api_key="s3cr3t"
        )

    def test_custom_default_model_stored_in_config(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        conn = AsyncConnection("http://localhost:6333", default_model="BAAI/bge-small-en-v1.5")
        assert conn.config.default_model == "BAAI/bge-small-en-v1.5"

    def test_config_and_executor_properties_return_correct_types(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        conn = AsyncConnection("http://localhost:6333")
        assert isinstance(conn.config, QQLConfig)
        assert isinstance(conn.executor, AsyncExecutor)


# ── TestAsyncConnectionRunQuery ────────────────────────────────────────────────

@pytest.mark.anyio
class TestAsyncConnectionRunQuery:
    """AsyncConnection.run_query() pipes through the Lexer → Parser → AsyncExecutor."""

    async def test_run_query_calls_executor_execute(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection("http://localhost:6333")
        await conn.run_query("SHOW COLLECTIONS")
        mock_executor.execute.assert_called_once()

    async def test_executor_instance_reused_across_queries(self, mocker):
        """AsyncExecutor() is constructed once; run_query() never re-instantiates it."""
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="ok", data=[]
        )
        executor_cls = mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection("http://localhost:6333")
        await conn.run_query("SHOW COLLECTIONS")
        await conn.run_query("SHOW COLLECTIONS")
        await conn.run_query("SHOW COLLECTIONS")

        # AsyncExecutor constructor called exactly once, not once per query
        executor_cls.assert_called_once()
        # But execute() called three times
        assert mock_executor.execute.call_count == 3

    async def test_invalid_query_raises_qql_syntax_error(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        conn = AsyncConnection("http://localhost:6333")
        with pytest.raises(QQLSyntaxError):
            await conn.run_query("TOTALLY INVALID QUERY GIBBERISH")

    async def test_run_query_returns_execution_result(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True, message="1 collection(s) found", data=["docs"]
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection("http://localhost:6333")
        result = await conn.run_query("SHOW COLLECTIONS")
        assert isinstance(result, ExecutionResult)
        assert result.success is True


# ── TestAsyncConnectionLifecycle ───────────────────────────────────────────────

@pytest.mark.anyio
class TestAsyncConnectionLifecycle:
    """AsyncConnection.close() and the async context-manager protocol."""

    async def test_close_calls_client_close(self, mocker):
        mock_client = AsyncMock()
        mocker.patch("qql.async_connection.AsyncQdrantClient", return_value=mock_client)
        conn = AsyncConnection("http://localhost:6333")
        await conn.close()
        mock_client.close.assert_called_once()

    async def test_context_manager_closes_on_exit(self, mocker):
        mock_client = AsyncMock()
        mocker.patch("qql.async_connection.AsyncQdrantClient", return_value=mock_client)

        async with AsyncConnection("http://localhost:6333") as conn:
            assert conn._client is mock_client

        mock_client.close.assert_called_once()

    async def test_context_manager_closes_on_exception(self, mocker):
        mock_client = AsyncMock()
        mocker.patch("qql.async_connection.AsyncQdrantClient", return_value=mock_client)

        with pytest.raises(RuntimeError, match="oops"):
            async with AsyncConnection("http://localhost:6333"):
                raise RuntimeError("oops")

        mock_client.close.assert_called_once()


# ── TestAsyncConnectionBatch ───────────────────────────────────────────────────

@pytest.mark.anyio
class TestAsyncConnectionBatch:
    """AsyncConnection batching support (run_queries_batch, run_parameterized_batch, QQLAsyncBatch)."""

    async def test_run_queries_batch(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            message="Batch executed successfully",
            data=[
                ExecutionResult(success=True, message="Found 1 result(s)"),
                ExecutionResult(success=True, message="Found 2 result(s)"),
            ],
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection()
        results = await conn.run_queries_batch([
            "SEARCH docs SIMILAR TO 'neurology' LIMIT 5",
            "SEARCH docs SIMILAR TO 'cardiology' LIMIT 5",
        ])
        assert len(results) == 2
        assert results[0].message == "Found 1 result(s)"
        assert results[1].message == "Found 2 result(s)"

    async def test_run_parameterized_batch(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            message="Batch executed successfully",
            data=[
                ExecutionResult(success=True, message="ok"),
                ExecutionResult(success=True, message="ok"),
            ],
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection()
        results = await conn.run_parameterized_batch(
            "SEARCH docs SIMILAR TO :query LIMIT 5 WHERE category = :category",
            [
                {"query": "brain stroke", "category": "Neurology"},
                {"query": "heart attack", "category": "Cardiology"},
            ],
        )
        assert len(results) == 2
        mock_executor.execute.assert_called_once()
        stmt = mock_executor.execute.call_args[0][0]
        # Verify both statements compiled correctly
        assert len(stmt.statements) == 2

    async def test_run_parameterized_query(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            message="ok",
            data="res1",
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection()
        result = await conn.run_parameterized_query(
            "SEARCH docs SIMILAR TO :query LIMIT 5 WHERE category = :category",
            {"query": "brain stroke", "category": "Neurology"},
        )

        stmt = mock_executor.execute.call_args[0][0]
        assert result.data == "res1"
        assert stmt.query_text == "brain stroke"

    async def test_qql_async_batch_context_manager(self, mocker):
        mocker.patch("qql.async_connection.AsyncQdrantClient")
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            message="Batch executed successfully",
            data=[
                ExecutionResult(success=True, message="Res 1", data="d1"),
                ExecutionResult(success=True, message="Res 2", data="d2"),
            ],
        )
        mocker.patch("qql.async_connection.AsyncExecutor", return_value=mock_executor)

        conn = AsyncConnection()
        async with QQLAsyncBatch(conn) as batch:
            ref1 = batch.add("SEARCH docs SIMILAR TO 'neurology' LIMIT 5")
            ref2 = batch.add("SEARCH docs SIMILAR TO 'cardiology' LIMIT 5")

        assert ref1.result.message == "Res 1"
        assert ref2.result.message == "Res 2"
        assert ref1.result.data == "d1"
        assert ref2.result.data == "d2"


# ── TestArchitecturalGapsClosed ────────────────────────────────────────────────

@pytest.mark.anyio
class TestArchitecturalGapsClosed:
    """Rigorous tests covering async execution, race conditions, strict parser validation, and ID propagation."""

    async def test_async_search_embeds_once(self, mocker):
        """AsyncExecutor keeps the hot path direct and avoids threadpool overhead for cached embeddings."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        
        # Mock embedders to track how they are called
        mocker.patch("qql.async_executor.Embedder.__init__", return_value=None)
        mock_embed = mocker.patch("qql.async_executor.Embedder.embed", return_value=[0.1, 0.2])
        
        from qql import QQLConfig
        executor = AsyncExecutor(mock_client, QQLConfig(url="http://localhost:6333"))
        
        from qql.parser import Parser
        from qql.lexer import Lexer
        
        node = Parser(Lexer().tokenize("SEARCH docs SIMILAR TO 'neurology' LIMIT 5")).parse()
        
        result = await executor.execute(node)
        assert result.success is True
        mock_embed.assert_called_once_with("neurology")

    async def test_batch_parsing_rejects_trailing_statements(self):
        """run_queries_batch must raise QQLSyntaxError when a query contains trailing tokens."""
        conn = AsyncConnection()
        with pytest.raises(QQLSyntaxError, match="Expected EOF"):
            await conn.run_queries_batch([
                "SHOW COLLECTIONS; DROP COLLECTION x"
            ])

    async def test_batched_insert_propagates_correct_ids(self, mocker):
        """_execute_batch_block preserves individual point IDs when aggregating Inserts into a bulk Insert."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.upsert.return_value = None
        
        # Mock get_collection to return a mock config with matching vector size
        mock_info = mocker.MagicMock()
        mock_info.config.params.vectors.size = 2
        mock_client.get_collection.return_value = mock_info
        
        mocker.patch("qql.async_executor.Embedder.__init__", return_value=None)
        mocker.patch("qql.async_executor.Embedder.embed", return_value=[0.1, 0.2])
        
        from qql.executor import CollectionTopology
        topology = CollectionTopology(
            exists=True,
            is_named_dense=False,
            has_unnamed_dense=True,
            dense_names=(),
            sparse_names=(),
        )
        mocker.patch("qql.async_executor.AsyncExecutor._resolve_topology", return_value=topology)
        
        from qql import QQLConfig
        executor = AsyncExecutor(mock_client, QQLConfig(url="http://localhost:6333"))
        
        from qql.parser import Parser
        from qql.lexer import Lexer
        
        # Aggregate multiple INSERTs inside a Batch block statement
        qql_batch = (
            "BEGIN BATCH\n"
            "INSERT INTO COLLECTION docs VALUES {'text': 'a', 'id': 101};\n"
            "INSERT INTO COLLECTION docs VALUES {'text': 'b', 'id': 102};\n"
            "END BATCH"
        )
        node = Parser(Lexer().tokenize(qql_batch)).parse()
        res = await executor.execute(node)
        
        assert res.success is True
        # Check that individual execution results correctly maintain their original custom point ID identity!
        assert len(res.data) == 2
        assert res.data[0].data["id"] == 101
        assert res.data[1].data["id"] == 102

    async def test_race_condition_collection_creation(self, mocker):
        """Concurrent inserts into a non-existent collection serialize creation to avoid Qdrant conflicts."""
        import asyncio
        mock_client = AsyncMock()
        
        # Mock get_collection to return a mock config with matching vector size
        mock_info = mocker.MagicMock()
        mock_info.config.params.vectors.size = 2
        mock_client.get_collection.return_value = mock_info
        
        from qql.executor import CollectionTopology
        # Mock resolve_topology sequence using real CollectionTopology objects
        topology_sequence = [
            CollectionTopology(exists=False, is_named_dense=False), # First insert task resolve topology
            CollectionTopology(exists=False, is_named_dense=False), # Second insert task resolve topology
            CollectionTopology(exists=False, is_named_dense=False), # Inside lock for first insert
            CollectionTopology(exists=True, is_named_dense=False, has_unnamed_dense=True, dense_names=(), sparse_names=()), # Inside lock for second insert
        ]
        
        mocker.patch("qql.async_executor.Embedder.__init__", return_value=None)
        mocker.patch("qql.async_executor.Embedder.embed", return_value=[0.1, 0.2])
        
        # Override _resolve_topology to yield the sequence
        calls = 0
        async def mock_resolve(*args, **kwargs):
            nonlocal calls
            val = topology_sequence[calls]
            calls += 1
            return val
            
        mocker.patch("qql.async_executor.AsyncExecutor._resolve_topology", side_effect=mock_resolve)
        mocker.patch("qql.async_executor.AsyncExecutor._create_collection_and_wait", return_value=None)
        
        from qql import QQLConfig
        executor = AsyncExecutor(mock_client, QQLConfig(url="http://localhost:6333"))
        
        from qql.parser import Parser
        from qql.lexer import Lexer
        
        insert_node_1 = Parser(Lexer().tokenize("INSERT INTO COLLECTION docs VALUES {'text': 'a', 'id': 1}")).parse()
        insert_node_2 = Parser(Lexer().tokenize("INSERT INTO COLLECTION docs VALUES {'text': 'b', 'id': 2}")).parse()
        
        # Fire both concurrently
        res1, res2 = await asyncio.gather(
            executor.execute(insert_node_1),
            executor.execute(insert_node_2),
        )
        
        assert res1.success is True
        assert res2.success is True
        # Verify that _create_collection_and_wait was called exactly once despite concurrency!
        executor._create_collection_and_wait.assert_called_once()

    def test_strict_batch_grammar(self):
        """Parser must raise QQLSyntaxError if a batch block ends with bare END instead of END BATCH."""
        from qql.parser import Parser
        from qql.lexer import Lexer
        
        invalid_batch = (
            "BEGIN BATCH\n"
            "SHOW COLLECTIONS\n"
            "END"
        )
        with pytest.raises(QQLSyntaxError, match="Expected BATCH"):
            Parser(Lexer().tokenize(invalid_batch)).parse()
