from __future__ import annotations

from typing import Any
from qdrant_client import AsyncQdrantClient

from .config import DEFAULT_MODEL, QQLConfig
from .async_executor import AsyncExecutor
from .executor import ExecutionResult
from .lexer import Lexer
from .parser import Parser


class AsyncConnection:
    """Stateful asynchronous connection to a Qdrant instance.

    Creates a single ``AsyncQdrantClient`` and ``AsyncExecutor`` once and reuses
    them for every :meth:`run_query` call — more efficient than the standalone
    one-shot helpers, which create a fresh client on every invocation.

    **Basic usage**::

        conn = AsyncConnection("http://localhost:6333", secret="my-key")
        result = await conn.run_query(
            "INSERT INTO COLLECTION docs VALUES {'text': 'hello world'}"
        )
        result = await conn.run_query("SEARCH docs SIMILAR TO 'hello' LIMIT 5")
        await conn.close()

    **Context manager (preferred)**::

        async with AsyncConnection("http://localhost:6333") as conn:
            result = await conn.run_query("SHOW COLLECTIONS")
            print(result.data)

    **Qdrant Cloud**::

        async with AsyncConnection("https://<cluster>.qdrant.io", secret="<api-key>") as conn:
            result = await conn.run_query("SHOW COLLECTIONS")

    **Custom embedding model**::

        async with AsyncConnection(
            "http://localhost:6333",
            default_model="BAAI/bge-base-en-v1.5",
        ) as conn:
            result = await conn.run_query(
                "INSERT INTO COLLECTION docs VALUES {'text': 'hello'}"
            )
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        secret: str | None = None,
        default_model: str | None = None,
        verify: bool | str = True,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
    ) -> None:
        """Create an asynchronous connection to a Qdrant instance.

        Args:
            url: Base URL of the Qdrant instance (default: ``http://localhost:6333``).
            secret: API key for authenticated instances; ``None`` for unauthenticated.
            default_model: Dense embedding model used when no ``USING MODEL`` clause
                is specified. Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
            verify: SSL certificate verification. Set to ``False`` to skip
                verification for self-signed/internal certificates, or pass
                a path to a custom CA bundle (default: ``True``).
            prefer_grpc: Whether to connect via fast gRPC transport.
            grpc_port: The gRPC port of Qdrant instance (default: 6334).
        """
        self._config = QQLConfig(
            url=url,
            secret=secret,
            default_model=default_model or DEFAULT_MODEL,
            verify=verify,
        )
        client_kwargs = {"url": url, "api_key": secret, "verify": verify}
        if prefer_grpc:
            client_kwargs["prefer_grpc"] = True
            client_kwargs["grpc_port"] = grpc_port
        self._client = AsyncQdrantClient(**client_kwargs)
        self._executor = AsyncExecutor(self._client, self._config)

    # ── Public API ────────────────────────────────────────────────────────

    async def run_query(self, query: str) -> ExecutionResult:
        """Parse and execute a single QQL statement asynchronously.

        Args:
            query: A QQL query string, e.g. ``"SEARCH docs SIMILAR TO 'hello' LIMIT 5"``.

        Returns:
            An :class:`~qql.ExecutionResult` with ``success``, ``message``, and ``data`` fields.

        Raises:
            QQLSyntaxError: The query string could not be parsed.
            QQLRuntimeError: The query parsed correctly but Qdrant rejected it.
        """
        tokens = Lexer().tokenize(query)
        node = Parser(tokens).parse()
        return await self._executor.execute(node)

    async def close(self) -> None:
        """Close the underlying Qdrant asynchronous connection pool."""
        await self._client.close()

    # ── Context manager ───────────────────────────────────────────────────

    async def __aenter__(self) -> AsyncConnection:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    # ── Power-user properties ─────────────────────────────────────────────

    @property
    def config(self) -> QQLConfig:
        """The :class:`~qql.QQLConfig` in use (url, secret, default_model)."""
        return self._config

    @property
    def executor(self) -> AsyncExecutor:
        """Direct access to the :class:`~qql.AsyncExecutor` for low-level use.

        Example — run multiple statements sharing a pre-built AST node::

            from qql.lexer import Lexer
            from qql.parser import Parser

            conn = AsyncConnection("http://localhost:6333")
            tokens = Lexer().tokenize("SHOW COLLECTIONS")
            node = Parser(tokens).parse()
            result = await conn.executor.execute(node)
        """
        return self._executor
