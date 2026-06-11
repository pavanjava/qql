from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qql-cli")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .config import (
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_MODEL,
    DEFAULT_SPARSE_VECTOR_NAME,
    QQLConfig,
    load_config,
)
from .connection import Connection
from .exceptions import QQLError, QQLRuntimeError, QQLSyntaxError
from .executor import ExecutionResult, Executor
from .lexer import Lexer
from .parser import Parser

__all__ = [
    "__version__",
    "Connection",
    "DEFAULT_DENSE_VECTOR_NAME",
    "DEFAULT_MODEL",
    "DEFAULT_SPARSE_VECTOR_NAME",
    "QQLConfig",
    "QQLError",
    "QQLRuntimeError",
    "QQLSyntaxError",
    "ExecutionResult",
    "Executor",
    "Lexer",
    "Parser",
    "load_config",
    "run_query",
]


def run_query(
    query: str,
    url: str = "http://localhost:6333",
    secret: str | None = None,
    default_model: str | None = None,
    verify: bool | str = True,
    prefer_grpc: bool = False,
    grpc_port: int = 6334,
) -> ExecutionResult:
    """One-shot convenience function kept for backward compatibility.

    Creates a :class:`Connection`, runs one query, closes the connection, and
    returns the result.  The underlying ``QdrantClient`` is always released —
    even if the query raises — so repeated calls do not leak resources.

    For workloads that issue multiple queries, prefer :class:`Connection`
    directly — it reuses a single client across all calls::

        with Connection(url, secret=secret) as conn:
            result = conn.run_query(query)
    """
    with Connection(
        url=url,
        secret=secret,
        default_model=default_model,
        verify=verify,
        prefer_grpc=prefer_grpc,
        grpc_port=grpc_port,
    ) as conn:
        return conn.run_query(query)
