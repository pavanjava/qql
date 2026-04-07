from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qql-cli")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .config import DEFAULT_MODEL, QQLConfig, load_config
from .exceptions import QQLError, QQLRuntimeError, QQLSyntaxError
from .executor import ExecutionResult, Executor
from .lexer import Lexer
from .parser import Parser

__all__ = [
    "__version__",
    "QQLConfig",
    "QQLError",
    "QQLRuntimeError",
    "QQLSyntaxError",
    "ExecutionResult",
    "Executor",
    "Lexer",
    "Parser",
    "run_query",
]


def run_query(
    query: str,
    url: str = "http://localhost:6333",
    secret: str | None = None,
    default_model: str | None = None,
) -> ExecutionResult:
    """Convenience function for programmatic use."""
    from qdrant_client import QdrantClient

    cfg = QQLConfig(
        url=url,
        secret=secret,
        default_model=default_model or DEFAULT_MODEL,
    )
    client = QdrantClient(url=url, api_key=secret)
    tokens = Lexer().tokenize(query)
    node = Parser(tokens).parse()
    return Executor(client, cfg).execute(node)
