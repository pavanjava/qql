# Reference — Models, Config, Project Structure, Errors

---

## Embedding Models

QQL uses [Fastembed](https://github.com/qdrant/fastembed) to convert text into vectors locally — no external API call is needed.

### Dense embedding (default)

```
sentence-transformers/all-MiniLM-L6-v2
```

- Vector dimensions: **384**
- Size: ~90 MB (downloaded on first use, cached locally)
- Good balance of speed and quality for English text

### Sparse embedding (hybrid mode default)

```
Qdrant/bm25
```

- Classic BM25 with IDF weighting
- Indices and values are generated as a sparse vector; no fixed dimensions
- Uses asymmetric encoding: `embed()` for documents, `query_embed()` for queries

### Specifying models

```sql
-- Dense only with custom model
INSERT INTO docs VALUES {'text': 'hello'} USING MODEL 'BAAI/bge-small-en-v1.5'
SEARCH docs SIMILAR TO 'hello' LIMIT 5 USING MODEL 'BAAI/bge-small-en-v1.5'

-- Hybrid with custom dense model
SEARCH docs SIMILAR TO 'hello' LIMIT 5 USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'

-- Hybrid with explicit fusion strategy
SEARCH docs SIMILAR TO 'hello' LIMIT 5 USING HYBRID FUSION 'dbsf'

-- Hybrid with both custom
SEARCH docs SIMILAR TO 'hello' LIMIT 5
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'prithivida/Splade_PP_en_v1'
```

### Commonly available dense models (Fastembed)

| Model | Dimensions | Notes |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default. Fast, good general quality |
| `BAAI/bge-small-en-v1.5` | 384 | Strong English retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Higher quality, larger size |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality, slowest |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Strong semantic similarity |

### Commonly available sparse models (Fastembed)

| Model | Notes |
|---|---|
| `Qdrant/bm25` | Default sparse model. Classic BM25 + IDF |
| `prithivida/Splade_PP_en_v1` | SPLADE++ — strong keyword + semantic overlap |
| `Qdrant/Unicoil` | UniCOIL sparse encoder |

### Cross-encoder reranking (RERANK default)

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

| Model | Notes |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Default. Fast passage reranker |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Larger, higher quality |
| `BAAI/bge-reranker-base` | Strong general-purpose reranker |
| `BAAI/bge-reranker-large` | Highest quality, slower |

> Models are downloaded automatically on first use and cached by Fastembed.

### Model consistency rule

Every collection is created with a fixed vector size determined by the model used on first INSERT. If you try to INSERT using a different model that produces different dimensions, QQL raises:

```
Error: Vector dimension mismatch: collection 'docs' expects 384 dims,
but model produces 768 dims. Specify a compatible model with USING MODEL '<model>'.
```

---

## Value Types in Dictionaries

| Type | Example | Notes |
|---|---|---|
| String | `'hello'` or `"hello"` | Single or double quotes |
| Integer | `42`, `-7` | Whole numbers, negative allowed |
| Float | `3.14`, `-0.5` | Decimal numbers |
| Boolean | `true`, `false` | Case-insensitive |
| Null | `null` | Case-insensitive |
| Nested dict | `{'key': 'val'}` | Arbitrary nesting |
| List | `['a', 'b', 1]` | Mixed types allowed |

Trailing commas in dicts and lists are allowed.

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
| `default_model` | Dense embedding model used when no `USING MODEL` clause is given |

You can edit this file directly to change the default model without reconnecting.

---

## Project Structure

```
qql/
├── pyproject.toml          # Package config; installs the `qql` CLI command
├── src/
│   └── qql/
│       ├── __init__.py     # Public API: run_query()
│       ├── cli.py          # CLI entry point: connect, disconnect, execute, dump, REPL
│       ├── config.py       # QQLConfig dataclass + ~/.qql/config.json I/O
│       ├── exceptions.py   # QQLError, QQLSyntaxError, QQLRuntimeError
│       ├── lexer.py        # Tokenizer: string → List[Token]
│       ├── ast_nodes.py    # Frozen dataclasses for each statement and filter type
│       ├── parser.py       # Recursive descent parser: tokens → AST node
│       ├── embedder.py     # Embedder (dense) + SparseEmbedder (BM25) + CrossEncoderEmbedder (rerank)
│       ├── executor.py     # AST node → Qdrant client call + filter + hybrid search
│       ├── script.py       # Script runner: parse and execute .qql files statement by statement
│       └── dumper.py       # Collection exporter: scroll all points → .qql INSERT BULK script
└── tests/
    ├── test_lexer.py       # Tokenizer unit tests
    ├── test_parser.py      # Parser unit tests
    ├── test_executor.py    # Executor unit tests (mocked Qdrant client)
    ├── test_script.py      # Script runner unit tests
    └── test_dumper.py      # Dumper unit tests
```

---

## Running Tests

Tests do not require a running Qdrant instance — the Qdrant client is mocked.

```bash
pytest tests/ -v
```

Expected output: **405 tests passing**.

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
| `Qdrant error during SEARCH: ...` | Hybrid search on a non-hybrid collection, or wrong vector names | Ensure the collection was created with `HYBRID` before using `USING HYBRID` in INSERT/SEARCH |
| `Unknown index type '...'` | Invalid schema type in CREATE INDEX | Use one of: `keyword`, `integer`, `float`, `bool`, `text`, `geo`, `datetime` |
| `Qdrant error during CREATE INDEX: ...` | Qdrant rejected the index creation | Check field name and collection state |
