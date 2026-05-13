# QQL — Qdrant Query Language

> SQL-like query language and CLI for [Qdrant](https://qdrant.tech) vector database.

[![PyPI version](https://img.shields.io/pypi/v/qql-cli?color=blue&label=PyPI)](https://pypi.org/project/qql-cli/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/qql-cli)](https://pypi.org/project/qql-cli/)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-375%20passing-brightgreen)](tests/)

Write `INSERT`, `SEARCH`, `SCROLL`, `RECOMMEND`, `DELETE`, and `CREATE COLLECTION` statements instead of Python SDK calls. Supports hybrid dense+sparse vector search, cross-encoder reranking, quantization (scalar, turbo, binary, product), SQL-style `WHERE` filters, script execution, and collection dump/restore.

```
qql> INSERT INTO COLLECTION notes VALUES {'text': 'Qdrant is a vector database', 'author': 'alice', 'year': 2024}
✓ Inserted 1 point [3f2e1a4b-8c91-4d0e-b123-abc123def456]

qql> SEARCH notes SIMILAR TO 'vector storage engines' LIMIT 3 WHERE year >= 2023
✓ Found 1 result(s)
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼──────────────────────────────────────
 0.8931 │ 3f2e1a4b-8c91-4d0e-b123-abc123def456 │ {'text': 'Qdrant is a ...', 'author': 'alice', 'year': 2024}

qql> SEARCH notes SIMILAR TO 'vector databases' LIMIT 5 USING HYBRID RERANK
✓ Found 1 result(s) (hybrid, reranked)
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼──────────────────────────────────────
 5.3754 │ 3f2e1a4b-8c91-4d0e-b123-abc123def456 │ {'text': 'Qdrant is a ...', 'author': 'alice', 'year': 2024}
```

---

## How It Works

QQL is a thin translation layer between a SQL-like query language and the Qdrant Python client. Every statement you type goes through three stages:

```
Your query string
      │
      ▼
  [ Lexer ]      — tokenizes the input into keywords, identifiers, literals
      │
      ▼
  [ Parser ]     — builds a typed AST node (e.g. InsertStmt, SearchStmt)
      │
      ▼
  [ Executor ]   — maps the AST node to a Qdrant client call
      │
      ▼
  Qdrant instance
```

When you run `INSERT`, the `text` field is automatically converted into a dense vector using [Fastembed](https://github.com/qdrant/fastembed). In **hybrid mode** (`USING HYBRID`), a sparse BM25 vector is also generated alongside the dense vector, and searches use Qdrant's Reciprocal Rank Fusion (RRF) to merge the results of both retrieval methods.

---

## Installation

**Requirements:** Python 3.12+, a running Qdrant instance.

```bash
pip install qql-cli
```

Connect to a Qdrant instance:

```bash
# Local
qql connect --url http://localhost:6333

# Qdrant Cloud
qql connect --url https://<your-cluster>.qdrant.io --secret <your-api-key>
```

Then type `qql` to open the interactive shell.

---

## Documentation

Full documentation lives in the [`docs/`](docs/) folder and at **[pavanjava.github.io/qql](https://pavanjava.github.io/qql)**:

| Topic | Description |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation, connecting, first queries |
| [INSERT / INSERT BULK](docs/insert.md) | Adding documents, batch inserts, payload types |
| [SEARCH / SCROLL / RECOMMEND / Hybrid / RERANK](docs/search.md) | Semantic search, pagination, hybrid, reranking, recommendations |
| [WHERE Filters](docs/filters.md) | Full SQL-style filter operators |
| [Collections & Quantization](docs/collections.md) | CREATE, DROP, QUANTIZE (scalar/turbo/binary/product), CREATE INDEX |
| [Scripts: EXECUTE / DUMP](docs/scripts.md) | Script files, collection backup/restore |
| [Programmatic Usage](docs/programmatic.md) | Use QQL as a Python library |
| [Reference: Models / Config / Errors](docs/reference.md) | Embedding models, config file, error reference |

---

## Quick Syntax Reference

```sql
-- Insert
INSERT INTO COLLECTION articles VALUES {'text': '...', 'year': 2024}
INSERT BULK INTO COLLECTION articles VALUES [{'text': '...'}, {'text': '...'}]

-- Search
SEARCH articles SIMILAR TO 'query' LIMIT 10
SEARCH articles SIMILAR TO 'query' LIMIT 10 WHERE year >= 2020
SEARCH articles SIMILAR TO 'query' LIMIT 10 USING HYBRID
SEARCH articles SIMILAR TO 'query' LIMIT 10 USING HYBRID RERANK

-- Scroll
SCROLL FROM articles LIMIT 50
SCROLL FROM articles WHERE year >= 2024 LIMIT 50
SCROLL FROM articles AFTER 'cursor-id' LIMIT 50

-- Recommend
RECOMMEND FROM articles POSITIVE IDS (1001, 1002) LIMIT 5

-- Collections
CREATE COLLECTION articles
CREATE COLLECTION articles HYBRID
CREATE COLLECTION articles QUANTIZE SCALAR
CREATE COLLECTION articles QUANTIZE TURBO
CREATE COLLECTION articles QUANTIZE TURBO BITS 2
CREATE COLLECTION articles QUANTIZE TURBO BITS 1.5 ALWAYS RAM
CREATE INDEX ON COLLECTION articles FOR year TYPE integer
SHOW COLLECTIONS
DROP COLLECTION articles

-- Delete
DELETE FROM articles WHERE id = '3f2e1a4b-...'
DELETE FROM articles WHERE year < 2020

-- Scripts
EXECUTE /path/to/script.qql
DUMP articles /path/to/backup.qql
```

---

## Running Tests

Tests do not require a running Qdrant instance — the Qdrant client is mocked.

```bash
pytest tests/ -v
```

Expected: **375 tests passing**.

---

## License

MIT © [Kameshwara Pavan Kumar Mantha](https://github.com/pavanjava)
