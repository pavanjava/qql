# SHOW COLLECTIONS

List available Qdrant collections.

## Syntax

```sql
SHOW COLLECTIONS
```

## Example

```sql
SHOW COLLECTIONS
```

## Output

Human-readable output lists collection names.

Programmatic execution returns a list of names in `ExecutionResult.data`.

## Limitations

- This statement lists names only.
- Use the planned `DESCRIBE COLLECTION` statement, once implemented, for vector config, point counts, and index details.

