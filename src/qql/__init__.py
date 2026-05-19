from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qql-cli")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .config import DEFAULT_MODEL, QQLConfig, load_config
from .connection import Connection
from .exceptions import QQLError, QQLRuntimeError, QQLSyntaxError
from .executor import ExecutionResult, Executor
from .lexer import Lexer
from .parser import Parser

__all__ = [
    "__version__",
    "Connection",
    "DEFAULT_MODEL",
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
    with Connection(url=url, secret=secret, default_model=default_model) as conn:
        return conn.run_query(query)
