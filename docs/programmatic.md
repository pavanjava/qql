# Programmatic Usage

QQL can be used as a Python library without the CLI.

---

## `run_query()` — high-level API

```python
from qql import run_query

# Insert a document (dense-only)
result = run_query(
    "INSERT INTO COLLECTION notes VALUES {'text': 'hello world', 'author': 'alice', 'year': 2024}",
    url="http://localhost:6333",
)
print(result.message)   # "Inserted 1 point [<id>]"
print(result.data)      # {"id": 1001 or "<uuid>", "collection": "notes"}

# Insert with hybrid vectors
result = run_query(
    "INSERT INTO COLLECTION notes VALUES {'text': 'hello world'} USING HYBRID",
    url="http://localhost:6333",
)
print(result.message)   # "Inserted 1 point [<id>] (hybrid)"

# Dense search with WHERE filter
result = run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5 WHERE year >= 2023 AND author != 'bot'",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["payload"])

# Hybrid search with WHERE filter
result = run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5 USING HYBRID WHERE year >= 2023",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["payload"])

# Scroll / pagination
result = run_query(
    "SCROLL FROM notes LIMIT 2",
    url="http://localhost:6333",
)
for point in result.data["points"]:
    print(point["id"], point["payload"])
print(result.data["next_offset"])

# Bulk insert (all records embedded and upserted in one call)
result = run_query(
    """INSERT BULK INTO COLLECTION notes VALUES [
      {'id': 1, 'text': 'first document', 'year': 2023},
      {'id': 2, 'text': 'second document', 'year': 2024}
    ]""",
    url="http://localhost:6333",
)
print(result.message)   # "Inserted 2 points"

# Recommend similar points using known IDs as positive examples
result = run_query(
    "RECOMMEND FROM notes POSITIVE IDS (1, 2) NEGATIVE IDS (3) LIMIT 5",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["payload"])

# Retrieve a point by ID
result = run_query(
    "SELECT * FROM notes WHERE id = 1",
    url="http://localhost:6333",
)
print(result.data)      # {"id": "1", "payload": {...}}

# Delete by filter
result = run_query(
    "DELETE FROM notes WHERE year < 2023",
    url="http://localhost:6333",
)
print(result.message)   # "Deleted N point(s)"

# Inspect collection diagnostics
result = run_query(
    "SHOW COLLECTION notes",
    url="http://localhost:6333",
)
print(result.data["topology"])         # "dense" or "hybrid"
print(result.data["vectors"])          # {"": {...}} or {"dense": {...}, ...}
print(result.data["payload_schema"])   # {"field": "keyword", ...} or None
```

---

## Low-level pipeline API

For more control, use the pipeline directly:

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

---

## ExecutionResult

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
| SCROLL | `{"points": [{"id": str, "payload": dict}, ...], "next_offset": str \| None}` |
| RECOMMEND | `[{"id": str, "score": float, "payload": dict}, ...]` |
| SHOW COLLECTIONS | `["name1", "name2", ...]` |
| SHOW COLLECTION | `{"name": str, "status": str, "points_count": int \| None, "indexed_vectors_count": int \| None, "segments_count": int, "topology": str, "vectors": dict, "sparse_vectors": dict \| None, "quantization": str \| None, "hnsw_config": dict, "payload_schema": dict \| None, "sharding": dict}` |
| CREATE COLLECTION | `None` |
| CREATE INDEX | `None` |
| DROP COLLECTION | `None` |
| DELETE | `None` |
