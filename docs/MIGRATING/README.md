# Migrating to QQL

This directory is for practical migration notes from raw Qdrant SDK calls or other vector database workflows into QQL.

Keep migration guides honest: QQL is useful for readable CLI/script workflows, but it is not a replacement for every SDK feature.

## Planned Guides

| Guide | Purpose | Status |
|---|---|---|
| `python-sdk-to-qql.md` | Map common `qdrant-client` operations to QQL | Planned |
| `rest-api-to-qql.md` | Convert common curl examples to QQL | Planned |
| `go-sdk-to-qql.md` | Map common Go client operations to QQL | Planned |
| `sql-to-qql.md` | Explain where SQL instincts transfer and where they do not | Planned |

## Quick Examples

### Insert

Python SDK:

```python
client.upsert(
    collection_name="articles",
    points=[PointStruct(id=1, vector=[...], payload={"text": "Hello"})],
)
```

QQL:

```sql
INSERT INTO COLLECTION articles VALUES {'id': 1, 'text': 'Hello'}
```

### Search With Filter

Python SDK:

```python
client.query_points(
    collection_name="articles",
    query=[...],
    query_filter=Filter(...),
    limit=5,
)
```

QQL:

```sql
SEARCH articles SIMILAR TO 'machine learning' LIMIT 5 WHERE category = 'ml'
```

## Migration Checklist

- Map collections and payload fields.
- Decide whether each collection should be dense-only or hybrid.
- Create payload indexes for fields used in filters.
- Convert a small sample first.
- Compare search results before migrating a full workflow.
- Keep raw SDK code where QQL does not yet expose the needed Qdrant feature.

## Adding a Guide

Each guide should include:

- what is being migrated from
- side-by-side examples
- limitations
- performance or indexing notes
- tested setup
