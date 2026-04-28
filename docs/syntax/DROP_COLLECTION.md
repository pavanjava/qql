# DROP COLLECTION

Delete a Qdrant collection.

## Syntax

```sql
DROP COLLECTION <name>
```

## Example

```sql
DROP COLLECTION articles
```

## Behavior

- The collection must exist.
- The operation deletes the collection and its points.

## Limitations

- There is no confirmation prompt inside QQL syntax.
- Collection snapshots and restore workflows are not exposed through QQL yet.

