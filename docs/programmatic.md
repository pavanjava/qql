---
layout: default
title: "Programmatic Usage"
---

# Programmatic Usage

QQL can be used as a Python library without the CLI.

---

## `Connection` — Primary API

`Connection` is the recommended way to use QQL programmatically. It opens a
single connection to Qdrant once and reuses it for every `run_query()` call —
more efficient than the legacy `run_query()` function, which creates a new
client on every invocation.

### Basic usage

```python
from qql import Connection

conn = Connection("http://localhost:6333")

# Insert a document (dense-only)
result = conn.run_query(
    "INSERT INTO COLLECTION notes VALUES {'text': 'hello world', 'author': 'alice', 'year': 2024}"
)
print(result.message)   # "Inserted 1 point [<id>]"
print(result.data)      # {"id": "<uuid>", "collection": "notes"}

# Search
result = conn.run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5 SCORE THRESHOLD 0.8 WHERE year >= 2023"
)
for hit in result.data:
    print(hit["score"], hit["payload"])

conn.close()
```

### Context manager (preferred)

The context manager guarantees the HTTP connection pool is released even if an
exception occurs:

```python
from qql import Connection

with Connection("http://localhost:6333") as conn:
    # All queries share the same connection
    conn.run_query(
        "INSERT INTO COLLECTION notes VALUES {'text': 'hello world'} USING HYBRID"
    )
    result = conn.run_query(
        "SEARCH notes SIMILAR TO 'hello' LIMIT 5 USING HYBRID WHERE year >= 2023"
    )
    for hit in result.data:
        print(hit["score"], hit["payload"])
```

### Qdrant Cloud

```python
from qql import Connection

with Connection("https://<your-cluster>.qdrant.io", secret="<your-api-key>") as conn:
    result = conn.run_query("SHOW COLLECTIONS")
    print(result.data)
```

### Custom embedding model

```python
from qql import Connection

with Connection(
    "http://localhost:6333",
    default_model="BAAI/bge-base-en-v1.5",
) as conn:
    conn.run_query(
        "INSERT INTO COLLECTION articles VALUES {'text': 'Attention is all you need'}"
    )
```

### All statement examples

```python
from qql import Connection

with Connection("http://localhost:6333") as conn:

    # Hybrid insert
    conn.run_query(
        "INSERT INTO COLLECTION notes VALUES {'text': 'hello world'} USING HYBRID"
    )

    # Dense search with WHERE filter
    result = conn.run_query(
        "SEARCH notes SIMILAR TO 'hello' LIMIT 5 WHERE year >= 2023 AND author != 'bot'"
    )
    for hit in result.data:
        print(hit["score"], hit["payload"])

    # Hybrid search
    result = conn.run_query(
        "SEARCH notes SIMILAR TO 'hello' LIMIT 5 USING HYBRID WHERE year >= 2023"
    )

    # Scroll / pagination
    result = conn.run_query("SCROLL FROM notes LIMIT 2")
    for point in result.data["points"]:
        print(point["id"], point["payload"])
    next_cursor = result.data["next_offset"]   # str | int | None

    # Continue pagination
    if next_cursor is not None:
        result = conn.run_query(f"SCROLL FROM notes AFTER '{next_cursor}' LIMIT 2")

    # Bulk insert
    result = conn.run_query(
        """INSERT BULK INTO COLLECTION notes VALUES [
          {'id': 1, 'text': 'first document', 'year': 2023},
          {'id': 2, 'text': 'second document', 'year': 2024}
        ]"""
    )
    print(result.message)   # "Inserted 2 points"

    # Recommend similar points
    result = conn.run_query(
        "RECOMMEND FROM notes POSITIVE IDS (1, 2) NEGATIVE IDS (3) LIMIT 5 SCORE THRESHOLD 0.6"
    )
    for hit in result.data:
        print(hit["score"], hit["payload"])

    # Retrieve a point by ID
    result = conn.run_query("SELECT * FROM notes WHERE id = 1")
    print(result.data)      # {"id": "1", "payload": {...}}

    # Delete by filter
    conn.run_query("DELETE FROM notes WHERE year < 2023")

    # Inspect collection diagnostics
    result = conn.run_query("SHOW COLLECTION notes")
    print(result.data["topology"])         # "dense" or "hybrid"
    print(result.data["vectors"])          # named vectors, or {"": {...}} for unnamed external collections
    print(result.data["payload_schema"])   # field index info, or None
```

### `Connection` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | `"http://localhost:6333"` | Qdrant instance URL |
| `secret` | `str \| None` | `None` | API key; `None` for unauthenticated |
| `default_model` | `str \| None` | `None` → `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model used when no `USING MODEL` clause is given |
| `default_dense_vector_name` | `str` | `"dense"` | Dense vector name used when QQL creates a collection and no explicit `USING VECTOR` name is given |
| `default_sparse_vector_name` | `str` | `"sparse"` | Sparse vector name used when QQL creates a hybrid collection and no explicit sparse vector name is given |

### Power-user: `executor` property

For low-level access to the pipeline, use `conn.executor` directly:

```python
from qql import Connection
from qql.lexer import Lexer
from qql.parser import Parser

with Connection("http://localhost:6333") as conn:
    tokens = Lexer().tokenize("SEARCH docs SIMILAR TO 'hello' LIMIT 5")
    node = Parser(tokens).parse()
    result = conn.executor.execute(node)
```

---

## `run_query()` — Legacy one-shot API

> **Note:** `run_query()` is kept for backward compatibility. It creates a new
> `Connection` (and therefore a new `QdrantClient`) on every call. For
> workloads that issue more than one query, use `Connection` instead.

```python
from qql import run_query

# Insert a document
result = run_query(
    "INSERT INTO COLLECTION notes VALUES {'text': 'hello world', 'author': 'alice', 'year': 2024}",
    url="http://localhost:6333",
)
print(result.message)

# Search
result = run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5 WHERE year >= 2023",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["payload"])
```

`run_query()` accepts the same `url`, `secret`, and `default_model` parameters
as `Connection.__init__()`.

---

## Low-level pipeline API

For full control, use the Lexer → Parser → Executor pipeline directly:

```python
from qdrant_client import QdrantClient
from qql.lexer import Lexer
from qql.parser import Parser
from qql.executor import Executor
from qql.config import QQLConfig

client = QdrantClient(url="http://localhost:6333")
config = QQLConfig(url="http://localhost:6333")
executor = Executor(client, config)

query = "SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 USING HYBRID WHERE category = 'cv'"
tokens = Lexer().tokenize(query)
node = Parser(tokens).parse()
result = executor.execute(node)

for hit in result.data:
    print(hit["score"], hit["payload"])
```

This is equivalent to what `Connection` does internally, giving you full
control over the client lifecycle and config.

---

## `ExecutionResult`

All operations return an `ExecutionResult`:

```python
@dataclass
class ExecutionResult:
    success: bool       # True if operation succeeded
    message: str        # Human-readable summary
    data: Any           # Operation-specific payload (see below)
```

| Operation | `result.data` type |
|---|---|
| INSERT (dense) | `{"id": int \| "<uuid>", "collection": "<name>"}` |
| INSERT (hybrid) | `{"id": int \| "<uuid>", "collection": "<name>"}` |
| INSERT BULK | `None` (count in `result.message`) |
| SELECT | `{"id": str, "payload": dict}` or `None` when not found |
| SEARCH | `[{"id": str, "score": float, "payload": dict}, ...]` |
| SCROLL | `{"points": [{"id": str, "payload": dict}, ...], "next_offset": str \| int \| None}` |
| RECOMMEND | `[{"id": str, "score": float, "payload": dict}, ...]` |
| SHOW COLLECTIONS | `["name1", "name2", ...]` |
| SHOW COLLECTION | `{"name": str, "status": str, "points_count": int \| None, "indexed_vectors_count": int \| None, "segments_count": int, "topology": str, "vectors": dict, "sparse_vectors": dict \| None, "quantization": str \| None, "hnsw_config": dict, "payload_schema": dict \| None, "sharding": dict}` |
| CREATE COLLECTION | `None` |
| CREATE INDEX | `None` |
| DROP COLLECTION | `None` |
| DELETE | `None` |
