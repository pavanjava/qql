# QQL Syntax Reference

This page is a compact index of the QQL language surface. It lists implemented syntax and planned syntax separately so the docs do not imply pages or features exist before they do.

For a complete narrative walkthrough, see the root `README.md`. For focused syntax details, use the pages linked below.

## Implemented Statements

These statements are parsed by QQL and executed against Qdrant.

| Statement | Description | Python `qql-cli` |
|---|---|:---:|
| [`CREATE COLLECTION`](CREATE_COLLECTION.md) | Create dense or hybrid collections | Supported |
| [`CREATE INDEX`](CREATE_INDEX.md) | Create a payload index | Supported |
| [`DROP COLLECTION`](DROP_COLLECTION.md) | Delete a collection | Supported |
| [`SHOW COLLECTIONS`](SHOW_COLLECTIONS.md) | List collections | Supported |
| [`INSERT`](INSERT.md) | Insert one point | Supported |
| [`INSERT BULK`](INSERT_BULK.md) | Insert multiple points | Supported |
| [`SEARCH`](SEARCH.md) | Dense, hybrid, sparse, filtered, and reranked search | Supported |
| [`RECOMMEND`](RECOMMEND.md) | Recommend by example IDs | Supported |
| [`DELETE`](DELETE.md) | Delete by ID or filter | Supported |

## Planned Statements

These are roadmap items, not current syntax.

| Statement | Purpose |
|---|---|
| `GET FROM <collection> WHERE id = '<id>'` | Retrieve a point by ID |
| `SCROLL <collection> LIMIT <n>` | Iterate through points |
| `COUNT <collection> WHERE <filter>` | Count points |
| `DESCRIBE COLLECTION <name>` | Show collection configuration and statistics |
| `UPDATE <collection> SET {...} WHERE id = '<id>'` | Update payload fields |
| `DELETE PAYLOAD <field> FROM <collection> WHERE id = '<id>'` | Remove payload keys |

## CLI / REPL Commands

These commands are handled by the CLI or REPL instead of the language parser.

| Command | Description | Python `qql-cli` |
|---|---|:---:|
| `qql connect --url <url>` | Save connection settings | Supported |
| `qql disconnect` | Remove saved connection settings | Supported |
| `qql execute <script.qql>` | Run a `.qql` script file | Supported |
| `DUMP COLLECTION <name> TO '<file.qql>'` | Export a collection to QQL statements | Supported |

## Clauses and Modifiers

| Clause | Used in | Description |
|---|---|---|
| `WHERE` | `SEARCH`, `RECOMMEND`, `DELETE` | Payload filtering |
| `USING MODEL '<model>'` | `CREATE COLLECTION`, `INSERT`, `SEARCH` | Pin dense embedding model |
| `USING HYBRID` | `CREATE COLLECTION`, `INSERT`, `SEARCH` | Use dense+sparse vectors |
| `DENSE MODEL '<model>'` | Hybrid `CREATE COLLECTION`, `INSERT`, `SEARCH` | Pin dense model |
| `SPARSE MODEL '<model>'` | Hybrid/sparse `INSERT`, `SEARCH` | Pin sparse model |
| `USING SPARSE` | `SEARCH` | Search sparse vector only |
| `RERANK` | `SEARCH` | Apply cross-encoder reranking |
| `EXACT` | `SEARCH`, `RECOMMEND` | Use exact search |
| `WITH { hnsw_ef, exact, acorn }` | `SEARCH`, `RECOMMEND` | Query-time search params |
| `LIMIT <n>` | `SEARCH`, `RECOMMEND` | Max results |
| `OFFSET <n>` | `RECOMMEND` | Skip initial results |
| `SCORE THRESHOLD <f>` | `RECOMMEND` | Filter low-scoring recommendations |
| `STRATEGY '<strategy>'` | `RECOMMEND` | Recommendation strategy |
| `LOOKUP FROM <collection>` | `RECOMMEND` | Use examples from another collection |

## Filter Operators

| Operator | Example |
|---|---|
| `=` | `status = 'active'` |
| `!=` | `status != 'draft'` |
| `>` / `>=` | `year >= 2020` |
| `<` / `<=` | `score < 0.8` |
| `BETWEEN ... AND` | `year BETWEEN 2020 AND 2024` |
| `IN (...)` | `status IN ('a', 'b')` |
| `NOT IN (...)` | `status NOT IN ('x', 'y')` |
| `IS NULL` / `IS NOT NULL` | `reviewer IS NOT NULL` |
| `IS EMPTY` / `IS NOT EMPTY` | `tags IS NOT EMPTY` |
| `MATCH` | `title MATCH 'vector database'` |
| `MATCH ANY` | `title MATCH ANY 'embedding retrieval'` |
| `MATCH PHRASE` | `title MATCH PHRASE 'semantic search'` |
| `AND` / `OR` / `NOT` | `status = 'active' AND NOT archived = true` |

## Adding Syntax Docs

When adding a dedicated page for a statement:

1. Create `docs/syntax/STATEMENT_NAME.md`.
2. Include syntax, examples, output shape, and limitations.
3. Link it from this index only after the page exists.
4. Update `docs/COMPATIBILITY.md` if implementation status changes.
