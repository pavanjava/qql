# INSERT BULK

Insert multiple points in one statement.

## Syntax

```sql
INSERT BULK INTO COLLECTION <collection> VALUES [{<payload>}, ...]
INSERT BULK INTO COLLECTION <collection> VALUES [{<payload>}, ...] USING MODEL '<dense_model>'
INSERT BULK INTO COLLECTION <collection> VALUES [{<payload>}, ...] USING HYBRID
INSERT BULK INTO COLLECTION <collection> VALUES [{<payload>}, ...] USING HYBRID DENSE MODEL '<dense_model>' SPARSE MODEL '<sparse_model>'
```

## Example

```sql
INSERT BULK INTO COLLECTION articles VALUES [
  {'id': 1, 'text': 'Dense vectors capture semantic similarity', 'category': 'search'},
  {'id': 2, 'text': 'Sparse vectors help with keyword matching', 'category': 'search'}
] USING HYBRID
```

## Behavior

- Each item must be a dictionary.
- Each item must contain `text`.
- Explicit IDs follow the same rules as `INSERT`.
- The selected model and hybrid mode apply to all items.

## Limitations

- Very large imports should be split into manageable script files.
- QQL currently embeds text client-side before upsert.

