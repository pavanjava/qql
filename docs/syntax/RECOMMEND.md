# RECOMMEND

Find points similar to existing example point IDs.

## Syntax

```sql
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) LIMIT <n>
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) NEGATIVE IDS (<id>, ...) LIMIT <n>
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) STRATEGY '<strategy>' LIMIT <n>
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) LOOKUP FROM <collection> LIMIT <n>
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) LIMIT <n> WHERE <filter>
RECOMMEND FROM <collection> POSITIVE IDS (<id>, ...) LIMIT <n> WITH { exact: true, hnsw_ef: 128 }
```

## Examples

```sql
RECOMMEND FROM articles POSITIVE IDS (1) LIMIT 5
```

```sql
RECOMMEND FROM articles
POSITIVE IDS (1, 2)
NEGATIVE IDS (3)
STRATEGY 'best_score'
LIMIT 10
WHERE category = 'search'
```

## Options

| Clause | Purpose |
|---|---|
| `NEGATIVE IDS` | Push results away from examples |
| `STRATEGY` | Select Qdrant recommendation strategy |
| `LOOKUP FROM` | Use examples from another collection |
| `USING '<vector>'` | Choose target named vector |
| `OFFSET` | Skip initial recommendations |
| `SCORE THRESHOLD` | Exclude low-scoring results |
| `WITH` | Pass search params |

## Limitations

- Example IDs must already exist.
- The supported strategies are the strategies exposed by the Qdrant client.

