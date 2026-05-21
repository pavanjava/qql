---
layout: default
title: "Getting Started"
---

# Getting Started with QQL

QQL is a SQL-like query language and CLI for [Qdrant](https://qdrant.tech). Instead of writing Python SDK calls you write natural query statements to insert, search, manage, and delete vector data.

---

## How It Works

Every statement goes through three stages:

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

When you run `INSERT`, the `text` field is automatically converted into a dense vector using [Fastembed](https://github.com/qdrant/fastembed). In **hybrid mode** (`USING HYBRID`), a sparse BM25 vector is also generated alongside the dense vector, and searches use Qdrant's Reciprocal Rank Fusion (RRF) by default to merge the results of both retrieval methods. You can override that with `FUSION 'dbsf'` on hybrid searches.

---

## Installation

**Requirements:** Python 3.12+, a running Qdrant instance.

### From PyPI

```bash
pip install qql-cli
```

### From source (development)

```bash
git clone https://github.com/pavanjava/qql
cd qql
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

After installation the `qql` command is available globally in your terminal.

---

## Connecting to Qdrant

Before running any queries you must connect to a Qdrant instance. The connection config is saved to `~/.qql/config.json` and reused automatically in future sessions.

### Local Qdrant (no API key)

```bash
qql connect --url http://localhost:6333
```

### Qdrant Cloud (with API key)

```bash
qql connect --url https://<your-cluster>.qdrant.io --secret <your-api-key>
```

On success you will see:

```
Connecting to http://localhost:6333...
Connected. Config saved to ~/.qql/config.json

QQL Interactive Shell  •  http://localhost:6333
Type help for available commands or exit to quit.

qql>
```

### Starting Qdrant locally with Docker

If you do not have a Qdrant instance running yet:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Disconnecting

To remove the saved connection config:

```bash
qql disconnect
```

---

## The QQL Shell

Once connected, running `qql` alone (no arguments) reads the saved config and opens the interactive shell:

```bash
qql
```

Inside the shell:

| Input | Effect |
|---|---|
| A QQL statement | Executes it and prints the result |
| `help` or `?` or `\h` | Prints a reference of all available commands |
| `EXECUTE <path>` or `\e <path>` | Runs a `.qql` script file |
| `DUMP <name> <output.qql>` | Exports a collection to a script file |
| `exit` or `quit` or `\q` or `:q` | Exits the shell |
| Empty line / Enter | Ignored |
| Ctrl-D or Ctrl-C | Exits the shell |

All keywords are **case-insensitive** — `INSERT`, `insert`, and `Insert` all work.

---

## Your First Queries

```sql
-- Create a collection and insert a document
INSERT INTO COLLECTION notes VALUES {'text': 'Qdrant is a vector database', 'author': 'alice', 'year': 2024}

-- Search for similar documents
SEARCH notes SIMILAR TO 'vector storage engines' LIMIT 3

-- Filter results
SEARCH notes SIMILAR TO 'vector databases' LIMIT 5 WHERE year >= 2023

-- Browse with pagination
SCROLL FROM notes LIMIT 10

-- List all collections
SHOW COLLECTIONS

-- Inspect one collection's diagnostics
SHOW COLLECTION notes

-- Retrieve a point by ID
SELECT * FROM notes WHERE id = 1
```

---

## Next Steps

- [INSERT / INSERT BULK](insert.md) — adding documents
- [SEARCH / SELECT / SCROLL / RECOMMEND / Hybrid / RERANK](search.md) — querying
- [WHERE Filters](filters.md) — payload filtering
- [Collections & Quantization](collections.md) — managing collections
- [Scripts: EXECUTE / DUMP](scripts.md) — automating with script files
- [Embedding Models](reference.md#embedding-models) — model reference
