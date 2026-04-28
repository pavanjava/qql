# CREATE COLLECTION

Create a Qdrant collection sized for QQL's embedding model.

## Syntax

```sql
CREATE COLLECTION <name>
CREATE COLLECTION <name> USING MODEL '<dense_model>'
CREATE COLLECTION <name> HYBRID
CREATE COLLECTION <name> USING HYBRID
CREATE COLLECTION <name> USING HYBRID DENSE MODEL '<dense_model>'
```

## Examples

```sql
CREATE COLLECTION articles
CREATE COLLECTION articles USING MODEL 'BAAI/bge-base-en-v1.5'
CREATE COLLECTION articles HYBRID
```

## Behavior

- Dense collections store one dense vector per point.
- Hybrid collections store a named dense vector and a named sparse vector.
- The dense vector size is inferred from the configured or requested embedding model.
- Distance is currently cosine.

## Limitations

- Custom distance, HNSW config, quantization, and on-disk payload options are not exposed yet.
- Multivector collections are not exposed yet.

