# INSERT

Insert one point into a collection.

## Syntax

```sql
INSERT INTO COLLECTION <collection> VALUES {<payload>}
INSERT INTO COLLECTION <collection> VALUES {<payload>} USING MODEL '<dense_model>'
INSERT INTO COLLECTION <collection> VALUES {<payload>} USING HYBRID
INSERT INTO COLLECTION <collection> VALUES {<payload>} USING HYBRID DENSE MODEL '<dense_model>' SPARSE MODEL '<sparse_model>'
```

## Examples

```sql
INSERT INTO COLLECTION articles VALUES {'text': 'Qdrant is a vector database'}
```

```sql
INSERT INTO COLLECTION articles VALUES {
  'id': 1,
  'text': 'Hybrid search combines dense and sparse retrieval',
  'category': 'search'
} USING HYBRID
```

## Behavior

- `text` is required and is embedded automatically.
- If `id` is omitted, QQL generates a UUID.
- Explicit IDs may be unsigned integers or UUID strings.
- If the collection does not exist, QQL can auto-create it using the selected embedding mode.
- Hybrid inserts write both dense and sparse vectors.

## Limitations

- QQL does not accept precomputed vectors in `INSERT`.
- Updating only payload fields is not exposed yet; use the planned `UPDATE` syntax once implemented.

