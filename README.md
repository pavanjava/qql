# QQL — Qdrant Query Language

A SQL-like CLI for [Qdrant](https://qdrant.tech), a high-performance vector database. Instead of writing Python SDK calls, you write natural query statements to insert, search, manage, and delete vector data — including rich SQL-style `WHERE` filters.

```
qql> INSERT INTO COLLECTION notes VALUES {'text': 'Qdrant is a vector database', 'author': 'alice', 'year': 2024}
✓ Inserted 1 point [3f2e1a4b-8c91-4d0e-b123-abc123def456]

qql> SEARCH notes SIMILAR TO 'vector storage engines' LIMIT 3 WHERE year >= 2023
✓ Found 1 result(s)
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼──────────────────────────────────────
 0.8931 │ 3f2e1a4b-8c91-4d0e-b123-abc123def456 │ {'text': 'Qdrant is a ...', 'author': 'alice', 'year': 2024}
```

---

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Connecting to Qdrant](#connecting-to-qdrant)
- [The QQL Shell](#the-qql-shell)
- [All QQL Operations](#all-qql-operations)
  - [INSERT — add a point](#insert--add-a-point)
  - [SEARCH — find similar points](#search--find-similar-points)
  - [WHERE Clause Filters](#where-clause-filters)
  - [SHOW COLLECTIONS — list collections](#show-collections--list-collections)
  - [CREATE COLLECTION — create a collection](#create-collection--create-a-collection)
  - [DROP COLLECTION — delete a collection](#drop-collection--delete-a-collection)
  - [DELETE — remove a point](#delete--remove-a-point)
- [Embedding Models](#embedding-models)
- [Value Types in Dictionaries](#value-types-in-dictionaries)
- [Configuration File](#configuration-file)
- [Programmatic Usage](#programmatic-usage)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Error Reference](#error-reference)

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

When you run `INSERT`, the `text` field in your dictionary is automatically converted into a dense vector using [Fastembed](https://github.com/qdrant/fastembed). The vector and the rest of your fields (stored as payload) are then upserted into Qdrant together. You never have to manage vectors manually.

`SEARCH` also embeds your query text and finds the nearest vectors by cosine similarity. An optional `WHERE` clause lets you pre-filter the candidate set using any payload field before similarity ranking — exactly like a SQL `WHERE` on top of a vector search.

---

## Installation

**Requirements:** Python 3.12+, a running Qdrant instance.

### From PyPI

```bash
pip install qql-cli
```

### From source (development)

```bash
git clone <repo>
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
| `exit` or `quit` or `\q` or `:q` | Exits the shell |
| Empty line / Enter | Ignored |
| Ctrl-D or Ctrl-C | Exits the shell |

All keywords are **case-insensitive** — `INSERT`, `insert`, and `Insert` all work.

---

## All QQL Operations

### INSERT — add a point

Inserts a new document into a collection. The `text` field is **mandatory** — it is automatically vectorized and stored as the point's vector. All other fields become searchable payload (metadata).

If the collection does not exist yet, it is **created automatically** with the correct vector dimensions.

**Syntax:**
```
INSERT INTO COLLECTION <collection_name> VALUES {<dict>}
INSERT INTO COLLECTION <collection_name> VALUES {<dict>} USING MODEL '<model_name>'
```

**Examples:**

Minimal insert (text only):
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'Qdrant supports cosine similarity search'}
```

Insert with metadata:
```sql
INSERT INTO COLLECTION articles VALUES {
  'text': 'Neural networks learn representations from data',
  'author': 'alice',
  'category': 'ml',
  'year': 2024,
  'published': true
}
```

Insert with a specific embedding model:
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'} USING MODEL 'BAAI/bge-small-en-v1.5'
```

Insert with nested metadata and tags:
```sql
INSERT INTO COLLECTION articles VALUES {
  'text': 'Attention is all you need',
  'meta': {'venue': 'NeurIPS', 'citations': 50000},
  'tags': ['transformers', 'attention', 'nlp']
}
```

**What happens internally:**
1. The `text` value is embedded into a dense vector using the configured model.
2. A UUID is auto-generated as the point ID.
3. All fields (including `text`) are stored in the payload.
4. The point is upserted into Qdrant.

**Rules:**
- `text` is always required. Omitting it raises an error.
- A point ID (UUID) is generated automatically — you do not provide one.
- If the collection already exists with a different vector size (from a different model), an error is raised with a clear message.

---

### SEARCH — find similar points

Performs a **semantic similarity search**: your query text is embedded with the same model used during insert, then Qdrant finds the nearest vectors by cosine distance.

An optional `WHERE` clause filters the candidate set **before** similarity ranking so you only get results that match both the semantic query and the payload conditions.

**Syntax:**
```
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n>
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING MODEL '<model_name>'
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING MODEL '<model_name>'] WHERE <filter>
```

**Examples:**

Basic search, return top 5 results:
```sql
SEARCH articles SIMILAR TO 'machine learning algorithms' LIMIT 5
```

Search only papers published after 2020:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 WHERE year > 2020
```

Search within a specific category, excluding drafts:
```sql
SEARCH articles SIMILAR TO 'neural networks' LIMIT 5 WHERE category = 'ml' AND status != 'draft'
```

Search with a model override and a filter:
```sql
SEARCH articles SIMILAR TO 'embeddings' LIMIT 10 USING MODEL 'BAAI/bge-small-en-v1.5' WHERE year >= 2022
```

**Output:**

Results are displayed as a table with three columns:

```
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼──────────────────────────────────
 0.9241 │ 3f2e1a4b-...                          │ {'text': 'Neural networks...', 'author': 'alice'}
 0.8817 │ 7a1b2c3d-...                          │ {'text': 'Attention is all...', 'tags': [...]}
```

- **Score** — cosine similarity score between 0 and 1. Higher is more similar.
- **ID** — the UUID of the matching point.
- **Payload** — all fields stored alongside the vector.

**Important:** Use the same model for SEARCH as you used for INSERT. Mixing models produces meaningless scores because the vectors live in different spaces.

---

### WHERE Clause Filters

The `WHERE` clause lets you filter on any payload field using SQL-style predicates. All standard comparison, range, membership, null-check, and full-text operators are supported.

#### Equality and inequality

```sql
-- Exact match
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE category = 'paper'

-- Not equal
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE status != 'draft'
```

#### Range comparisons

```sql
-- Greater than / less than
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE score > 0.8
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE year < 2024

-- Greater than or equal / less than or equal
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE score >= 0.75
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE year <= 2023
```

#### BETWEEN … AND

```sql
-- Inclusive range (equivalent to year >= 2018 AND year <= 2023)
SEARCH articles SIMILAR TO 'history of ai' LIMIT 10 WHERE year BETWEEN 2018 AND 2023
```

#### IN and NOT IN

```sql
-- Field value must be one of the listed values
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE status IN ('published', 'reviewed')

-- Field value must not be any of the listed values
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE status NOT IN ('deleted', 'archived')

-- Trailing commas are allowed
SEARCH articles SIMILAR TO 'x' LIMIT 5 WHERE status IN ('a', 'b',)
```

#### IS NULL and IS NOT NULL

```sql
-- Points where the reviewer field is absent or explicitly null
SEARCH articles SIMILAR TO 'peer review' LIMIT 5 WHERE reviewer IS NULL

-- Points where reviewer is set to any non-null value
SEARCH articles SIMILAR TO 'peer review' LIMIT 5 WHERE reviewer IS NOT NULL
```

#### IS EMPTY and IS NOT EMPTY

```sql
-- Points where the tags list is empty
SEARCH articles SIMILAR TO 'untagged' LIMIT 5 WHERE tags IS EMPTY

-- Points where the tags list has at least one element
SEARCH articles SIMILAR TO 'categorized' LIMIT 5 WHERE tags IS NOT EMPTY
```

#### Full-text MATCH

```sql
-- All terms in the string must appear in the field (full-text index required)
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH 'vector database'

-- Any term in the string can match
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH ANY 'embedding retrieval'

-- The exact phrase must appear
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH PHRASE 'semantic search'
```

> Full-text MATCH requires a Qdrant full-text index on the field. Create one in the Qdrant dashboard or via the SDK before using MATCH filters.

#### AND, OR, NOT — logical operators

Operator precedence: `NOT` (highest) > `AND` > `OR` (lowest). Use parentheses to override.

```sql
-- AND: both conditions must be true
SEARCH articles SIMILAR TO 'nlp' LIMIT 10 WHERE category = 'paper' AND year >= 2020

-- OR: either condition can be true
SEARCH articles SIMILAR TO 'llm' LIMIT 10 WHERE source = 'arxiv' OR source = 'pubmed'

-- NOT: negate a condition
SEARCH articles SIMILAR TO 'benchmark' LIMIT 10 WHERE NOT status = 'draft'

-- Chained AND (three conditions, all must hold)
SEARCH articles SIMILAR TO 'deep learning' LIMIT 20
  WHERE year >= 2019 AND category = 'cv' AND status != 'retracted'

-- Parentheses to group OR inside AND
SEARCH articles SIMILAR TO 'conference paper' LIMIT 10
  WHERE (source = 'arxiv' OR source = 'ieee') AND year >= 2022

-- NOT on a parenthesized group
SEARCH articles SIMILAR TO 'x' LIMIT 5 WHERE NOT (status = 'draft' OR status = 'deleted')
```

#### Dot-notation for nested fields

Qdrant supports nested payload fields accessed with dot notation. Use the same path syntax in `WHERE`:

```sql
-- Filter on meta.source nested field
SEARCH articles SIMILAR TO 'wikipedia' LIMIT 5 WHERE meta.source = 'web'

-- Filter on a deeply nested array field
SEARCH cities SIMILAR TO 'large city' LIMIT 5 WHERE country.cities[].population > 1000000
```

#### Combined example

```sql
-- Semantic search over research papers:
-- must be from arxiv or IEEE, published 2020–2023, not retracted, with a reviewer assigned
SEARCH papers SIMILAR TO 'attention mechanism transformers' LIMIT 20
  WHERE (source = 'arxiv' OR source = 'ieee')
    AND year BETWEEN 2020 AND 2023
    AND status != 'retracted'
    AND reviewer IS NOT NULL
```

#### Full filter reference

| WHERE syntax | Description |
|---|---|
| `field = 'x'` | Exact match |
| `field != 'x'` | Not equal |
| `field > n` | Greater than |
| `field >= n` | Greater than or equal |
| `field < n` | Less than |
| `field <= n` | Less than or equal |
| `field BETWEEN a AND b` | Inclusive range |
| `field IN ('a', 'b')` | Value in list |
| `field NOT IN ('a', 'b')` | Value not in list |
| `field IS NULL` | Field absent or null |
| `field IS NOT NULL` | Field present and non-null |
| `field IS EMPTY` | Field is an empty list |
| `field IS NOT EMPTY` | Field is a non-empty list |
| `field MATCH 'text'` | All terms present (full-text) |
| `field MATCH ANY 'text'` | Any term present (full-text) |
| `field MATCH PHRASE 'text'` | Exact phrase present (full-text) |
| `A AND B` | Both conditions must hold |
| `A OR B` | Either condition must hold |
| `NOT A` | Condition must not hold |
| `(A OR B) AND C` | Parentheses for grouping |
| `meta.source = 'x'` | Dot-notation nested field |

---

### SHOW COLLECTIONS — list collections

Lists all collections in the connected Qdrant instance.

**Syntax:**
```
SHOW COLLECTIONS
```

**Example:**
```sql
SHOW COLLECTIONS
```

**Output:**
```
✓ 3 collection(s) found
┌──────────────────┐
│ Collection       │
├──────────────────┤
│ articles         │
│ notes            │
│ products         │
└──────────────────┘
```

---

### CREATE COLLECTION — create a collection

Explicitly creates a new empty collection. Collections are also created automatically on the first INSERT, so this command is optional — use it when you want to pre-create a collection before inserting data.

**Syntax:**
```
CREATE COLLECTION <collection_name>
```

**Example:**
```sql
CREATE COLLECTION research_papers
```

The collection is created using the **default embedding model's dimensions** (384 for `all-MiniLM-L6-v2`) with **cosine distance**.

If the collection already exists, the command succeeds with a message and does nothing.

---

### DROP COLLECTION — delete a collection

Permanently deletes a collection and **all points inside it**. This operation is irreversible.

**Syntax:**
```
DROP COLLECTION <collection_name>
```

**Example:**
```sql
DROP COLLECTION old_experiments
```

Raises an error if the collection does not exist.

---

### DELETE — remove a point

Deletes a single point from a collection by its ID. The point ID is the UUID returned by INSERT.

**Syntax:**
```
DELETE FROM <collection_name> WHERE id = '<point_id>'
DELETE FROM <collection_name> WHERE id = <integer_id>
```

**Examples:**

Delete by UUID string:
```sql
DELETE FROM articles WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456'
```

Delete by integer ID:
```sql
DELETE FROM articles WHERE id = 42
```

To find a point's ID, run a SEARCH first and copy the ID from the results table.

---

## Embedding Models

QQL uses [Fastembed](https://github.com/qdrant/fastembed) to convert text into vectors locally — no external API call is needed.

### Default model

```
sentence-transformers/all-MiniLM-L6-v2
```

- Vector dimensions: **384**
- Size: ~90 MB (downloaded on first use, cached locally)
- Good balance of speed and quality for English text

### Specifying a different model

Add `USING MODEL '<model_name>'` to INSERT or SEARCH:

```sql
INSERT INTO docs VALUES {'text': 'hello'} USING MODEL 'BAAI/bge-small-en-v1.5'
SEARCH docs SIMILAR TO 'hello' LIMIT 5 USING MODEL 'BAAI/bge-small-en-v1.5'
```

`USING MODEL` and `WHERE` can be combined:

```sql
SEARCH docs SIMILAR TO 'hello' LIMIT 5 USING MODEL 'BAAI/bge-small-en-v1.5' WHERE year >= 2022
```

### Commonly available Fastembed models

| Model | Dimensions | Notes |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default. Fast, good general quality |
| `BAAI/bge-small-en-v1.5` | 384 | Strong English retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Higher quality, larger size |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality, slowest |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Strong semantic similarity |

> Models are downloaded automatically on first use and cached by Fastembed. Loading a new model for the first time takes a few seconds.

### Model consistency rule

Every collection is created with a fixed vector size determined by the model used on first INSERT (or CREATE COLLECTION). If you try to INSERT into an existing collection using a different model that produces different dimensions, QQL will raise an error:

```
Error: Vector dimension mismatch: collection 'docs' expects 384 dims,
but model produces 768 dims. Specify a compatible model with USING MODEL '<model>'.
```

---

## Value Types in Dictionaries

The `VALUES` dictionary (and nested dicts) supports these types:

| Type | Example | Notes |
|---|---|---|
| String | `'hello'` or `"hello"` | Single or double quotes |
| Integer | `42`, `-7` | Whole numbers, negative allowed |
| Float | `3.14`, `-0.5` | Decimal numbers |
| Boolean | `true`, `false` | Case-insensitive |
| Null | `null` | Case-insensitive |
| Nested dict | `{'key': 'val'}` | Arbitrary nesting |
| List | `['a', 'b', 1]` | Mixed types allowed |

**Example using every type:**
```sql
INSERT INTO demo VALUES {
  'text':    'example document',
  'count':   42,
  'score':   0.95,
  'active':  true,
  'deleted': false,
  'ref':     null,
  'meta':    {'source': 'web', 'lang': 'en'},
  'tags':    ['ai', 'nlp', 'search']
}
```

Trailing commas in dicts and lists are allowed:
```sql
INSERT INTO demo VALUES {'text': 'hi', 'x': 1,}
```

---

## Configuration File

The connection config is stored at `~/.qql/config.json`:

```json
{
  "url": "http://localhost:6333",
  "secret": null,
  "default_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

| Field | Description |
|---|---|
| `url` | Qdrant instance URL |
| `secret` | API key (null if not required) |
| `default_model` | Embedding model used when no `USING MODEL` clause is given |

You can edit this file directly to change the default model without reconnecting:

```json
{
  "url": "http://localhost:6333",
  "secret": null,
  "default_model": "BAAI/bge-small-en-v1.5"
}
```

---

## Programmatic Usage

QQL can also be used as a Python library without the CLI:

```python
from qql import run_query

# Insert a document
result = run_query(
    "INSERT INTO COLLECTION notes VALUES {'text': 'hello world', 'author': 'alice', 'year': 2024}",
    url="http://localhost:6333",
)
print(result.message)   # "Inserted 1 point [<uuid>]"
print(result.data)      # {"id": "...", "collection": "notes"}

# Basic search
result = run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["id"], hit["payload"])

# Search with a WHERE filter
result = run_query(
    "SEARCH notes SIMILAR TO 'hello' LIMIT 5 WHERE year >= 2023 AND author != 'bot'",
    url="http://localhost:6333",
)
for hit in result.data:
    print(hit["score"], hit["payload"])
```

Or use the pipeline directly for more control:

```python
from qdrant_client import QdrantClient
from qql.lexer import Lexer
from qql.parser import Parser
from qql.executor import Executor
from qql.config import QQLConfig

client = QdrantClient(url="http://localhost:6333")
config = QQLConfig(url="http://localhost:6333")
executor = Executor(client, config)

query = "SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 WHERE category = 'cv'"
tokens = Lexer().tokenize(query)
node = Parser(tokens).parse()
result = executor.execute(node)

for hit in result.data:
    print(hit["score"], hit["payload"])
```

### ExecutionResult

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
| INSERT | `{"id": "<uuid>", "collection": "<name>"}` |
| SEARCH | `[{"id": str, "score": float, "payload": dict}, ...]` |
| SHOW COLLECTIONS | `["name1", "name2", ...]` |
| CREATE COLLECTION | `None` |
| DROP COLLECTION | `None` |
| DELETE | `None` |

---

## Project Structure

```
qql/
├── pyproject.toml          # Package config; installs the `qql` CLI command
├── src/
│   └── qql/
│       ├── __init__.py     # Public API: run_query()
│       ├── cli.py          # CLI entry point: connect, disconnect, REPL
│       ├── config.py       # QQLConfig dataclass + ~/.qql/config.json I/O
│       ├── exceptions.py   # QQLError, QQLSyntaxError, QQLRuntimeError
│       ├── lexer.py        # Tokenizer: string → List[Token]
│       ├── ast_nodes.py    # Frozen dataclasses for each statement and filter type
│       ├── parser.py       # Recursive descent parser: tokens → AST node
│       ├── embedder.py     # Fastembed wrapper with per-model cache
│       └── executor.py     # AST node → Qdrant client call + filter conversion
└── tests/
    ├── test_lexer.py       # Tokenizer unit tests (keywords, operators, dot-paths)
    ├── test_parser.py      # Parser unit tests (all statements + WHERE filters)
    └── test_executor.py    # Executor unit tests (mocked Qdrant client + filter builders)
```

---

## Running Tests

Tests do not require a running Qdrant instance — the Qdrant client is mocked.

```bash
pytest tests/ -v
```

Expected output: **118 tests passing**.

---

## Error Reference

| Error | Cause | Fix |
|---|---|---|
| `Not connected. Run: qql connect --url <url>` | No `~/.qql/config.json` found | Run `qql connect --url <url>` first |
| `Connection failed: ...` | Qdrant unreachable at given URL | Check that Qdrant is running and the URL is correct |
| `INSERT requires a 'text' field in VALUES` | `text` key missing from the VALUES dict | Add `'text': '...'` to your dict |
| `Vector dimension mismatch: collection '...' expects X dims, but model produces Y dims` | Model used in INSERT differs from the one used to create the collection | Use `USING MODEL` to specify the same model as the collection was created with |
| `Collection '...' does not exist` | SEARCH / DROP / DELETE on a non-existent collection | Check name spelling or run `SHOW COLLECTIONS` |
| `Unexpected token '...'; expected a QQL statement keyword` | Unrecognized statement | Check the query syntax; QQL does not support SQL SELECT |
| `Unterminated string literal (at position N)` | A string is missing its closing quote | Close the string with a matching `'` or `"` |
| `Unexpected character '@' (at position N)` | A character not part of QQL syntax | Remove or quote the offending character |
| `Expected a filter operator after field '...'` | Unknown operator in WHERE clause | Use one of: `=`, `!=`, `>`, `>=`, `<`, `<=`, `IN`, `NOT IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`, `IS EMPTY`, `IS NOT EMPTY`, `MATCH` |
| `Expected ')' ...` | Unclosed parenthesis in WHERE clause | Add the missing `)` to close the group |
