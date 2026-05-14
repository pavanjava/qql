# Managing Collections

---

## SHOW COLLECTIONS — list collections

Lists all collections in the connected Qdrant instance.

```sql
SHOW COLLECTIONS
```

**Output:**
```
✓ 3 collection(s) found
┌──────────────────┐
│ Collection       │
├──────────────────┤
│ articles         │
│ notes            │
│ products         │
└──────────────────┘
```

---

## SHOW COLLECTION — inspect one collection

Returns collection diagnostics for a single collection using Qdrant's collection info.

**Syntax:**
```sql
SHOW COLLECTION <collection_name>
```

**What it shows:**

- Point count
- Indexed vector count
- Segment count
- Vector names, dimensions, and distance metrics
- Dense vs hybrid topology
- Sparse vector modifiers when present
- Quantization mode
- HNSW configuration
- Payload indexes detected by Qdrant
- Shard, replica, and write consistency settings

**Example:**
```sql
SHOW COLLECTION research_papers
```

**Output:**
```
OK Collection 'research_papers' diagnostics
Collection: research_papers
  Status               : green
  Points               : 12450
  Indexed vectors      : 12450
  Segments             : 3
  Topology             : hybrid
  Vector 'dense'       : 768 dims, Cosine distance
  Sparse 'sparse'      : modifier=idf
  Quantization         : scalar
  HNSW M               : 16
  HNSW ef_construct    : 100
  Payload indexes:
    category: keyword
    year: integer
  Shards               : 1
  Replicas             : 1
  Write consistency    : 1
```

**Notes:**

- `Topology` is `dense` for standard collections and `hybrid` when sparse vectors are configured alongside dense vectors.
- Dense collections with named vectors still report their vector names and dimensions.
- If no payload indexes exist, QQL prints `Payload indexes      : none`.
- Raises an error if the collection does not exist.

---

## CREATE COLLECTION — create a collection

Explicitly creates a new empty collection. Collections are also created automatically on the first INSERT, so this command is optional — use it when you want to pre-create a collection before inserting data.

**Syntax:**
```
CREATE COLLECTION <collection_name>
CREATE COLLECTION <collection_name> HYBRID
CREATE COLLECTION <collection_name> USING MODEL '<model_name>'
CREATE COLLECTION <collection_name> USING HYBRID
CREATE COLLECTION <collection_name> USING HYBRID DENSE MODEL '<model>'
```

Any of the above forms can be followed by an optional `QUANTIZE` clause — see [Quantization](#quantization--quantize-clause) below.

**Examples:**

Dense-only collection (standard, uses default model dimensions):
```sql
CREATE COLLECTION research_papers
```

Dense-only collection pinned to a specific model (768-dimensional):
```sql
CREATE COLLECTION research_papers USING MODEL 'BAAI/bge-base-en-v1.5'
```

Hybrid collection (dense + sparse BM25, default models):
```sql
CREATE COLLECTION research_papers HYBRID
```

Hybrid collection with a custom dense model:
```sql
CREATE COLLECTION research_papers USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
```

When `USING MODEL` is omitted, the collection uses the **default embedding model's dimensions** (384 for `all-MiniLM-L6-v2`). If the collection already exists, the command succeeds with a message and does nothing.

---

## Quantization — QUANTIZE clause

Quantization reduces the memory footprint of vector collections and speeds up search at the cost of a small, controllable accuracy loss. QQL supports all four Qdrant quantization strategies via an optional `QUANTIZE` clause appended to `CREATE COLLECTION`.

**Four strategies:**

| Type | Compression | Accuracy | Best For |
|---|---|---|---|
| `SCALAR` | 4× (float32 → int8) | < 1% loss | Most collections — best balance |
| `TURBO` | 8–32× (4-bit to 1-bit) | Low–medium | Better recall than BINARY at same storage budget |
| `BINARY` | 32× (float32 → 1-bit) | Higher loss | Speed priority; centered distributions only |
| `PRODUCT` | 4× (configurable) | Variable | Memory-constrained deployments |

**Full syntax:**
```
CREATE COLLECTION <name> ... QUANTIZE SCALAR [QUANTILE <0.0–1.0>] [ALWAYS RAM]
CREATE COLLECTION <name> ... QUANTIZE TURBO  [BITS <1|1.5|2|4>]   [ALWAYS RAM]
CREATE COLLECTION <name> ... QUANTIZE BINARY  [ALWAYS RAM]
CREATE COLLECTION <name> ... QUANTIZE PRODUCT [ALWAYS RAM]
```

- **`QUANTILE <float>`** — (SCALAR only) calibration quantile for the INT8 conversion; defaults to Qdrant's built-in default (0.99) when omitted.
- **`BITS <depth>`** — (TURBO only) bit depth passed to the Qdrant SDK:
  - `4` — 4-bit (default when `BITS` is omitted; server applies its own default)
  - `2` — 2-bit
  - `1.5` — 1.5-bit
  - `1` — 1-bit
  > Compression ratios (8×, 16×, 24×, 32×) and recall characteristics are
  > Qdrant server-side behaviors. QQL maps the `BITS` value to the SDK model and
  > passes it to Qdrant; actual results depend on your Qdrant server version.
- **`ALWAYS RAM`** — keep the **quantized** vectors in RAM at all times, regardless of the collection's `on_disk` setting. Improves search throughput at the cost of higher RAM usage for the compressed index. The original full-precision vectors are stored and managed independently of this flag. Supported by all four quantization types.
- **`QUANTIZE`** always appears **after** all other clauses (`HYBRID`, `USING MODEL`, etc.).
- For `PRODUCT`, the compression ratio is fixed at **4×** in this version.
- For `TURBO`, Cosine, Dot, and Euclidean distance are supported by the Qdrant server when TurboQuant is enabled.
- When used with `HYBRID` collections, quantization applies only to the **dense** vector.

**Examples:**

Scalar quantization (recommended default):
```sql
CREATE COLLECTION research_papers QUANTIZE SCALAR
```

Scalar with explicit calibration and quantized vectors pinned to RAM:
```sql
CREATE COLLECTION research_papers QUANTIZE SCALAR QUANTILE 0.95 ALWAYS RAM
```

TurboQuant — default 4-bit (8× compression, good recall):
```sql
CREATE COLLECTION research_papers QUANTIZE TURBO
```

TurboQuant — 2-bit (16× compression):
```sql
CREATE COLLECTION research_papers QUANTIZE TURBO BITS 2
```

TurboQuant — 1.5-bit (24× compression) with quantized vectors pinned to RAM:
```sql
CREATE COLLECTION research_papers QUANTIZE TURBO BITS 1.5 ALWAYS RAM
```

TurboQuant — 1-bit (32× compression, same ratio as BINARY but better recall):
```sql
CREATE COLLECTION research_papers QUANTIZE TURBO BITS 1
```

Binary quantization for large high-dimensional embeddings:
```sql
CREATE COLLECTION research_papers QUANTIZE BINARY
```

Product quantization for maximum memory savings:
```sql
CREATE COLLECTION research_papers QUANTIZE PRODUCT ALWAYS RAM
```

Combined with hybrid collection:
```sql
CREATE COLLECTION research_papers HYBRID QUANTIZE SCALAR
CREATE COLLECTION research_papers HYBRID QUANTIZE TURBO BITS 2
```

Combined with a pinned model:
```sql
CREATE COLLECTION research_papers USING MODEL 'BAAI/bge-base-en-v1.5' QUANTIZE SCALAR QUANTILE 0.99
CREATE COLLECTION research_papers USING MODEL 'BAAI/bge-base-en-v1.5' QUANTIZE TURBO BITS 2
```

Combined with hybrid + dense model:
```sql
CREATE COLLECTION research_papers USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' QUANTIZE TURBO
```

**Valid combinations:**

| Base form | + SCALAR | + TURBO | + BINARY | + PRODUCT |
|---|---|---|---|---|
| `CREATE COLLECTION name` | ✓ | ✓ | ✓ | ✓ |
| `... HYBRID` | ✓ | ✓ | ✓ | ✓ |
| `... USING MODEL 'x'` | ✓ | ✓ | ✓ | ✓ |
| `... USING HYBRID` | ✓ | ✓ | ✓ | ✓ |
| `... USING HYBRID DENSE MODEL 'x'` | ✓ | ✓ | ✓ | ✓ |

> INSERT and SEARCH on quantized collections work exactly the same as on non-quantized ones — no changes to INSERT or SEARCH syntax are needed.

---

## CREATE INDEX — create a payload index

Creates a payload index on a collection field. Payload indexes speed up `WHERE` clause filtering by allowing Qdrant to efficiently match on indexed fields.

**Syntax:**
```
CREATE INDEX ON COLLECTION <collection_name> FOR <field_name> TYPE <schema_type>
```

**Supported schema types:**

| Type | Description |
|---|---|
| `keyword` | Exact string match (e.g. status, category) |
| `integer` | Whole numbers |
| `float` | Decimal numbers |
| `bool` | Boolean values |
| `text` | Full-text search (enables `MATCH` operators) |
| `geo` | Geospatial coordinates |
| `datetime` | Date/time values |

**Examples:**

```sql
CREATE INDEX ON COLLECTION articles FOR category TYPE keyword
CREATE INDEX ON COLLECTION articles FOR year TYPE integer
CREATE INDEX ON COLLECTION articles FOR title TYPE text
CREATE INDEX ON COLLECTION articles FOR meta.author TYPE keyword
```

**Rules:**
- The collection must already exist. Raises an error otherwise.
- Indexes are idempotent — creating the same index twice succeeds silently.

---

## DROP COLLECTION — delete a collection

Permanently deletes a collection and **all points inside it**. This operation is irreversible.

```sql
DROP COLLECTION old_experiments
```

Raises an error if the collection does not exist.

---

## DELETE — remove points

Deletes one or more points from a collection by specific ID or by a `WHERE` filter.

**Syntax:**
```
DELETE FROM <collection_name> WHERE id = '<point_id>'
DELETE FROM <collection_name> WHERE id = <integer_id>
DELETE FROM <collection_name> WHERE <filter>
```

**Examples:**

```sql
-- Delete by UUID
DELETE FROM articles WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456'

-- Delete by integer ID
DELETE FROM articles WHERE id = 42

-- Delete all points matching a filter
DELETE FROM articles WHERE category = 'archived'

-- Delete with a compound filter
DELETE FROM articles WHERE year < 2020 AND status = 'draft'
```

**Notes:**
- If no points match the filter or ID, the operation succeeds silently with a count of 0.
- The collection itself must exist; deleting from a non-existent collection raises an error.
