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
CREATE COLLECTION <collection_name> WITH VECTORS { on_disk: <bool> }
CREATE COLLECTION <collection_name> WITH HNSW { m, ef_construct, full_scan_threshold, max_indexing_threads, on_disk, payload_m, inline_storage }
CREATE COLLECTION <collection_name> WITH OPTIMIZERS { deleted_threshold, vacuum_min_vector_number, default_segment_number, max_segment_size, memmap_threshold, indexing_threshold, flush_interval_sec, max_optimization_threads, prevent_unoptimized }
CREATE COLLECTION <collection_name> WITH PARAMS { replication_factor, write_consistency_factor, on_disk_payload }
ALTER COLLECTION <collection_name> WITH HNSW { ... }
ALTER COLLECTION <collection_name> WITH VECTORS { ... }
ALTER COLLECTION <collection_name> WITH OPTIMIZERS { ... }
ALTER COLLECTION <collection_name> WITH PARAMS { ... }
ALTER COLLECTION <collection_name> QUANTIZE SCALAR [QUANTILE <0.0–1.0>] [ALWAYS RAM]
ALTER COLLECTION <collection_name> QUANTIZE BINARY [ALWAYS RAM]
ALTER COLLECTION <collection_name> QUANTIZE PRODUCT [ALWAYS RAM]
ALTER COLLECTION <collection_name> QUANTIZE TURBO [BITS <1|1.5|2|4>] [ALWAYS RAM]
ALTER COLLECTION <collection_name> QUANTIZE DISABLED
```

Any `CREATE COLLECTION` form can be followed by an optional `QUANTIZE` clause and one or more `WITH ... { ... }` config blocks.

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

Dense collection with payload-aware HNSW links:
```sql
CREATE COLLECTION research_papers WITH HNSW { payload_m: 16 }
```

When `USING MODEL` is omitted, the collection uses the **default embedding model's dimensions** (384 for `all-MiniLM-L6-v2`). If the collection already exists, the command succeeds with a message and does nothing.

### Collection config blocks

QQL supports the same config blocks on both `CREATE COLLECTION` and `ALTER COLLECTION`:

- `WITH VECTORS { on_disk }`
- `WITH HNSW { m, ef_construct, full_scan_threshold, max_indexing_threads, on_disk, payload_m, inline_storage }`
- `WITH OPTIMIZERS { deleted_threshold, vacuum_min_vector_number, default_segment_number, max_segment_size, memmap_threshold, indexing_threshold, flush_interval_sec, max_optimization_threads, prevent_unoptimized }`
- `WITH PARAMS { replication_factor, write_consistency_factor, on_disk_payload }` on create
- `WITH PARAMS { replication_factor, write_consistency_factor, read_fan_out_factor, read_fan_out_delay_ms, on_disk_payload }` on alter
- `ALTER COLLECTION ... QUANTIZE ...` supports the same quantization forms as create, plus `QUANTIZE DISABLED`

Example:

```sql
CREATE COLLECTION tenant_docs USING HYBRID WITH HNSW { payload_m: 16, m: 32 }
ALTER COLLECTION tenant_docs WITH OPTIMIZERS { indexing_threshold: 10000 }
```

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
CREATE INDEX ON COLLECTION <collection_name> FOR <field_name> TYPE <schema_type> WITH { ... }
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
| `uuid` | UUID payload values |

**Examples:**

```sql
CREATE INDEX ON COLLECTION articles FOR category TYPE keyword
CREATE INDEX ON COLLECTION articles FOR tenant_id TYPE keyword WITH {is_tenant: true, on_disk: true, enable_hnsw: true}
CREATE INDEX ON COLLECTION articles FOR year TYPE integer
CREATE INDEX ON COLLECTION articles FOR doc_id TYPE uuid
CREATE INDEX ON COLLECTION articles FOR title TYPE text
CREATE INDEX ON COLLECTION articles FOR title TYPE text WITH {tokenizer: 'word', min_token_len: 2, max_token_len: 20, lowercase: true, phrase_matching: true}
CREATE INDEX ON COLLECTION articles FOR meta.author TYPE keyword
```

**Advanced options currently supported:**

- `keyword` / `uuid`
  - `is_tenant: true|false`
  - `on_disk: true|false`
  - `enable_hnsw: true|false`
- `text`
  - `tokenizer: 'prefix'|'whitespace'|'word'|'multilingual'`
  - `min_token_len: <int>`
  - `max_token_len: <int>`
  - `lowercase: true|false`
  - `ascii_folding: true|false`
  - `phrase_matching: true|false`
  - `stopwords: 'english'` or `stopwords: ['a', 'the']`
  - `on_disk: true|false`
  - `enable_hnsw: true|false`

**Rules:**
- The collection must already exist. Raises an error otherwise.
- Indexes are idempotent — creating the same index twice succeeds silently.
- Advanced `WITH { ... }` options are currently supported only for `keyword`, `uuid`, and `text`.

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

---

## UPDATE SET VECTOR — replace a point's dense vector

Replaces the stored dense vector for a **single point** identified by its ID. The point must already exist in the collection. Use this when you want to refresh an embedding without changing the payload.

**Syntax:**
```
UPDATE <collection> SET VECTOR WHERE id = '<point_id>' [<vector>]
UPDATE <collection> SET VECTOR WHERE id = <integer_id>  [<vector>]
```

The vector is provided as a JSON-style float array `[v1, v2, ..., vN]`. The array length must match the collection's configured vector dimensions.

**Examples:**

```sql
-- Replace vector by UUID
UPDATE articles SET VECTOR WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456' [0.1, 0.2, 0.3, 0.4]

-- Replace vector by integer ID
UPDATE articles SET VECTOR WHERE id = 42 [0.1, 0.2, 0.3, 0.4]
```

**Notes:**
- Only single-point updates are supported (by ID). Bulk or filter-based vector updates are not supported.
- The point must already exist; this operation does not create new points.
- The collection must exist; updating from a non-existent collection raises an error.
- For hybrid collections, the dense vector named `"dense"` is updated. Sparse vectors are managed separately.

---

## UPDATE SET PAYLOAD — merge fields into a point's payload

Merges new key/value pairs into the payload of one or more points. **Existing fields not mentioned in the update are preserved** (additive merge, not a full replace). Use a `WHERE` filter to update multiple points at once.

**Syntax:**
```
UPDATE <collection> SET PAYLOAD WHERE id = '<point_id>' {<payload>}
UPDATE <collection> SET PAYLOAD WHERE id = <integer_id>  {<payload>}
UPDATE <collection> SET PAYLOAD WHERE <filter>            {<payload>}
```

**Examples:**

```sql
-- Update a single point by UUID
UPDATE articles SET PAYLOAD WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456' {'year': 2025, 'status': 'active'}

-- Update a single point by integer ID
UPDATE articles SET PAYLOAD WHERE id = 42 {'category': 'tech'}

-- Update all points matching a filter
UPDATE articles SET PAYLOAD WHERE category = 'draft' {'status': 'published'}

-- Compound filter update
UPDATE articles SET PAYLOAD WHERE year < 2020 AND status = 'draft' {'archived': true}
```

**Notes:**
- **Merge semantics:** only the fields in `{…}` are written; all other existing payload fields are preserved.
- If no points match the filter, the operation succeeds silently with no changes.
- The collection must exist; updating from a non-existent collection raises an error.
- All `WHERE` filter operators supported by `DELETE` are also supported here (see [WHERE Filters](filters.md)).
