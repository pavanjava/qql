# CREATE INDEX

Create a Qdrant payload index for fields used in filters.

## Syntax

```sql
CREATE INDEX ON COLLECTION <collection> FOR <field> TYPE <type>
```

Supported Python index types:

| Type | Use for |
|---|---|
| `keyword` | exact string/category filters |
| `integer` | integer range/equality filters |
| `float` | floating-point range/equality filters |
| `bool` | boolean filters |
| `text` | full-text match filters |
| `geo` | geo payload fields |
| `datetime` | datetime payload fields |

## Examples

```sql
CREATE INDEX ON COLLECTION articles FOR category TYPE keyword
CREATE INDEX ON COLLECTION articles FOR year TYPE integer
CREATE INDEX ON COLLECTION articles FOR score TYPE float
```

## Behavior

- The collection must already exist.
- Dot notation is accepted for nested payload fields.
- Indexes improve filtered search performance in Qdrant.

## Limitations

- QQL does not currently expose advanced text index configuration.
- Companion implementation support should be checked before claiming cross-language parity for every index type.

