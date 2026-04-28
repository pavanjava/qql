# DELETE

Delete points by ID or by filter.

## Syntax

```sql
DELETE FROM <collection> WHERE id = '<uuid>'
DELETE FROM <collection> WHERE id = <integer>
DELETE FROM <collection> WHERE <filter>
```

## Examples

```sql
DELETE FROM articles WHERE id = 1
```

```sql
DELETE FROM articles WHERE status = 'archived'
```

## Behavior

- `WHERE id = ...` deletes a single point by ID.
- Any other supported filter deletes all matching points.
- The collection must exist.

## Limitations

- `DELETE PAYLOAD` is not implemented yet.
- There is no dry-run syntax yet. Use filters carefully.

