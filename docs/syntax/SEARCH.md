# SEARCH

Search a collection using dense, sparse, or hybrid retrieval.

## Syntax

```sql
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n>
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> WHERE <filter>
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> USING MODEL '<dense_model>'
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> USING HYBRID
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> USING SPARSE
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> RERANK
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> WITH { hnsw_ef: 128, exact: true, acorn: true }
```

## Examples

```sql
SEARCH articles SIMILAR TO 'vector search' LIMIT 5
```

```sql
SEARCH articles SIMILAR TO 'keyword heavy query' LIMIT 10 USING HYBRID WHERE category = 'search'
```

```sql
SEARCH articles SIMILAR TO 'high precision result' LIMIT 5 RERANK
```

## Filters

`WHERE` supports equality, inequality, ranges, `BETWEEN`, `IN`, `NOT IN`, null/empty checks, text match operators, and `AND`/`OR`/`NOT`.

```sql
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10
WHERE year >= 2023 AND title MATCH ANY 'hybrid sparse'
```

## Behavior

- Dense search is the default.
- `USING HYBRID` searches dense and sparse vectors and fuses results.
- `USING SPARSE` searches only the sparse vector.
- `RERANK` applies a cross-encoder reranking pass to retrieved candidates.
- `EXACT` is shorthand for exact search.

## Limitations

- `SEARCH ... OFFSET` is not implemented yet.
- Batch search is not implemented yet.
- Reranking adds latency and expects useful text in the result payload.

