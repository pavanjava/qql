"""Tests for the AsyncConnection class.

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
            url="https://cloud.example.io", api_key="s3cr3t", verify=True
        )

    def test_grpc_options_passed_to_async_qdrant_client(self, mocker):
        mock_client_cls = mocker.patch("qql.async_connection.AsyncQdrantClient")
        AsyncConnection(
            "http://localhost:6333",
            verify=False,
            prefer_grpc=True,
            grpc_port=9999,
        )
        mock_client_cls.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            verify=False,
            prefer_grpc=True,
            grpc_port=9999,
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
        mock_executor.execute.assert_awaited_once()

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
        assert mock_executor.execute.await_count == 3

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


# ── TestArchitecturalGapsClosed ────────────────────────────────────────────────

@pytest.mark.anyio
class TestArchitecturalGapsClosed:
    """Rigorous tests covering async execution and collection creation races."""

    async def test_async_topology_uses_single_get_collection_call(self, mocker):
        """Async topology resolution should mirror sync executor and avoid a separate exists call."""
        from qdrant_client.models import Distance, VectorParams

        mock_client = AsyncMock()
        mock_info = mocker.MagicMock()
        mock_info.config.params.vectors = VectorParams(size=2, distance=Distance.COSINE)
        mock_info.config.params.sparse_vectors = None
        mock_client.get_collection.return_value = mock_info

        executor = AsyncExecutor(mock_client, QQLConfig(url="http://localhost:6333"))
        topology = await executor._resolve_topology("docs")

        assert topology.exists is True
        assert topology.has_unnamed_dense is True
        mock_client.get_collection.assert_called_once_with("docs")
        mock_client.collection_exists.assert_not_called()

    async def test_async_insert_uses_raced_existing_unnamed_topology(self, mocker):
        """If another creator wins the race with an unnamed vector, send a plain vector payload."""
        mock_client = AsyncMock()
        mock_client.upsert.return_value = None

        from qql.executor import CollectionTopology

        topology_sequence = [
            CollectionTopology(exists=False, is_named_dense=False),
            CollectionTopology(
                exists=True,
                is_named_dense=False,
                has_unnamed_dense=True,
                dense_sizes=(("", 2),),
            ),
        ]

        mocker.patch("qql.async_executor.Embedder.__init__", return_value=None)
        mocker.patch("qql.async_executor.Embedder.embed", return_value=[0.1, 0.2])
        mocker.patch(
            "qql.async_executor.AsyncExecutor._resolve_topology",
            side_effect=topology_sequence,
        )
        create = mocker.patch(
            "qql.async_executor.AsyncExecutor._create_collection_and_wait",
            new_callable=AsyncMock,
        )

        executor = AsyncExecutor(mock_client, QQLConfig(url="http://localhost:6333"))

        from qql.parser import Parser
        from qql.lexer import Lexer

        node = Parser(
            Lexer().tokenize(
                "INSERT INTO COLLECTION docs VALUES {'text': 'a', 'id': 1}"
            )
        ).parse()
        result = await executor.execute(node)

        assert result.success is True
        create.assert_not_called()
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.vector == [0.1, 0.2]

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
