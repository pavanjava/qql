from __future__ import annotations

from typing import Any
from .config import DEFAULT_MODEL, QQLConfig
from .executor import Executor, ExecutionResult
from .lexer import Lexer
from .parser import Parser
from .utils import render_parameterized_query


class Connection:
    """Stateful connection to a Qdrant instance.

    Creates a single ``QdrantClient`` and ``Executor`` once and reuses them for
    every :meth:`run_query` call — more efficient than the standalone
    :func:`run_query` function, which creates a fresh client on every
    invocation.

    **Basic usage**::

        conn = Connection("http://localhost:6333", secret="my-key")
        result = conn.run_query(
            "INSERT INTO COLLECTION docs VALUES {'text': 'hello world'}"
        )
        result = conn.run_query("SEARCH docs SIMILAR TO 'hello' LIMIT 5")
        conn.close()

    **Context manager (preferred)** — the HTTP connection pool is always
    released, even if ``run_query`` raises::

        with Connection("http://localhost:6333") as conn:
            result = conn.run_query("SHOW COLLECTIONS")
            print(result.data)

    **Qdrant Cloud**::

        with Connection("https://<cluster>.qdrant.io", secret="<api-key>") as conn:
            result = conn.run_query("SHOW COLLECTIONS")

    **Custom embedding model**::

        with Connection(
            "http://localhost:6333",
            default_model="BAAI/bge-base-en-v1.5",
        ) as conn:
            result = conn.run_query(
                "INSERT INTO COLLECTION docs VALUES {'text': 'hello'}"
            )
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        secret: str | None = None,
        default_model: str | None = None,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
        verify: bool | str = True,
    ) -> None:
        """Create a connection to a Qdrant instance.

        Args:
            url: Base URL of the Qdrant instance (default: ``http://localhost:6333``).
            secret: API key for authenticated instances; ``None`` for unauthenticated.
            default_model: Dense embedding model used when no ``USING MODEL`` clause
                is specified.  Defaults to
                ``sentence-transformers/all-MiniLM-L6-v2``.
            prefer_grpc: Whether to connect via fast gRPC transport.
            grpc_port: The gRPC port of Qdrant instance (default: 6334).
            verify: SSL certificate verification. Set to ``False`` to skip
                verification for self-signed/internal certificates, or pass
                a path to a custom CA bundle (default: ``True``).
        """
        from qdrant_client import QdrantClient

        self._config = QQLConfig(
            url=url,
            secret=secret,
            default_model=default_model or DEFAULT_MODEL,
        )
        client_kwargs = {"url": url, "api_key": secret}
        if prefer_grpc:
            client_kwargs["prefer_grpc"] = True
            client_kwargs["grpc_port"] = grpc_port
        self._client = QdrantClient(**client_kwargs)
        self._client = QdrantClient(url=url, api_key=secret, verify=verify)
        self._executor = Executor(self._client, self._config)

    # ── Public API ────────────────────────────────────────────────────────

    def run_query(self, query: str) -> ExecutionResult:
        """Parse and execute a single QQL statement.

        Args:
            query: A QQL query string, e.g.
                ``"SEARCH docs SIMILAR TO 'hello' LIMIT 5"``.

        Returns:
            An :class:`~qql.ExecutionResult` with ``success``, ``message``,
            and ``data`` fields.

        Raises:
            QQLSyntaxError: The query string could not be parsed.
            QQLRuntimeError: The query parsed correctly but Qdrant rejected it.
        """
        tokens = Lexer().tokenize(query)
        node = Parser(tokens).parse()
        return self._executor.execute(node)

    def run_queries_batch(self, queries: list[str]) -> list[ExecutionResult]:
        """Parse and execute a batch of QQL statements.

        Combines compatible operations (such as SEARCH queries) to execute in
        a single network request.
        """
        from .ast_nodes import BatchBlockStmt
        nodes = []
        for q in queries:
            tokens = Lexer().tokenize(q)
            node = Parser(tokens).parse()
            nodes.append(node)

        batch_node = BatchBlockStmt(statements=tuple(nodes))
        res = self._executor.execute(batch_node)
        return res.data

    def run_parameterized_query(self, template: str, params: dict[str, Any]) -> ExecutionResult:
        """Execute one QQL query template with named parameters.

        Uses named placeholders prefixed with ':' (e.g. :query, :category).
        """
        return self.run_query(render_parameterized_query(template, params))

    def run_parameterized_batch(self, template: str, params: list[dict[str, Any]]) -> list[ExecutionResult]:
        """Execute a single QQL query template with a batch of parameters.

        Uses named placeholders prefixed with ':' (e.g. :query, :category).
        """
        queries = [render_parameterized_query(template, p) for p in params]
        return self.run_queries_batch(queries)

    def close(self) -> None:
        """Close the underlying Qdrant HTTP connection pool.

        Call this explicitly when not using the context-manager form, or let
        the ``with`` statement handle it automatically.
        """
        self._client.close()

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Power-user properties ─────────────────────────────────────────────

    @property
    def config(self) -> QQLConfig:
        """The :class:`~qql.QQLConfig` in use (url, secret, default_model)."""
        return self._config

    @property
    def executor(self) -> Executor:
        """Direct access to the :class:`~qql.Executor` for low-level use.

        Example — run multiple statements sharing a pre-built AST node::

            from qql.lexer import Lexer
            from qql.parser import Parser

            conn = Connection("http://localhost:6333")
            tokens = Lexer().tokenize("SHOW COLLECTIONS")
            node = Parser(tokens).parse()
            result = conn.executor.execute(node)
        """
        return self._executor


class QQLBatch:
    """Session context manager for executing batch queries and mutations in QQL."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._queries: list[str] = []
        self._proxies: list[OperationProxy] = []

    def add(self, query: str) -> OperationProxy:
        """Queue a QQL statement for batch execution."""
        self._queries.append(query)
        proxy = OperationProxy()
        self._proxies.append(proxy)
        return proxy

    def __enter__(self) -> QQLBatch:
        self._queries.clear()
        self._proxies.clear()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                return
            if not self._queries:
                return
            results = self.connection.run_queries_batch(self._queries)
            if len(results) != len(self._proxies):
                error = RuntimeError(
                    "Batch result count mismatch: "
                    f"expected {len(self._proxies)}, got {len(results)}"
                )
                for proxy in self._proxies:
                    proxy._reject(error)
                raise error
            for proxy, res in zip(self._proxies, results, strict=True):
                proxy._resolve(res)
        finally:
            self._queries.clear()
            self._proxies.clear()


class OperationProxy:
    """Proxy handle that resolves to an ExecutionResult after QQLBatch exits."""

    def __init__(self) -> None:
        self._result: ExecutionResult | None = None
        self._exception: RuntimeError | None = None

    def _resolve(self, result: ExecutionResult) -> None:
        self._result = result

    def _reject(self, exception: RuntimeError) -> None:
        self._exception = exception

    @property
    def result(self) -> ExecutionResult:
        """The resolved ExecutionResult."""
        if self._exception is not None:
            raise self._exception
        if self._result is None:
            raise RuntimeError("Batch has not been executed yet.")
        return self._result
