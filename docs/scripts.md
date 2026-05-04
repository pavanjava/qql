# Script Files — EXECUTE and DUMP

QQL supports reading from and writing to `.qql` script files, making it easy to automate bulk operations, seed databases, and back up collections.

---

## EXECUTE — run a .qql script file

Execute a file containing multiple QQL statements in sequence. Each statement is parsed and executed in order. `--` comments are stripped before parsing.

**CLI usage:**
```bash
qql execute /path/to/script.qql

# Stop on first error instead of continuing through all statements
qql execute /path/to/script.qql --stop-on-error
```

**In-shell usage (inside the QQL REPL):**
```
qql> EXECUTE /path/to/script.qql
qql> \e /path/to/script.qql
```

**Script format:**

```sql
-- This is a comment — the entire line is ignored
-- ============================================================
--  QQL Script — populate articles collection
-- ============================================================

-- Step 1: create the collection
CREATE COLLECTION articles

-- Step 2: bulk insert records
INSERT BULK INTO COLLECTION articles VALUES [
  {'text': 'Neural networks learn representations', 'year': 2023},
  {'text': 'Attention mechanisms in transformers',  'year': 2024}
]

-- Step 3: verify
SHOW COLLECTIONS
```

**Rules:**
- `--` to end-of-line is a comment and is ignored (inline or full-line)
- Statements can span multiple lines (e.g. `INSERT BULK ... VALUES [...]`)
- `RECOMMEND` statements work in `.qql` files the same way they do in the REPL
- Blank lines between statements are ignored
- By default all statements run even if one fails; use `--stop-on-error` to halt early

**Included examples:**
- [`resources/sample.qql`](../resources/sample.qql) seeds the demo medical dataset
- [`resources/sample_v2.qql`](../resources/sample_v2.qql) is a compact end-to-end example with explicit IDs and runnable `RECOMMEND` statements

**Example output:**
```
Executing: /path/to/script.qql

[1/3] CREATE COLLECTION articles
  ✓ Collection 'articles' created (384-dimensional vectors, cosine distance)
[2/3] INSERT BULK INTO COLLECTION articles VALUES [ …
  ✓ Inserted 2 points
[3/3] SHOW COLLECTIONS
  ✓ 1 collection(s) found

Done. 3/3 statement(s) succeeded.
```

---

## DUMP COLLECTION — export collection to a .qql script file

Export every point in a collection to a `.qql` script file. The generated file is valid QQL that re-creates the collection and re-inserts all payload data. Points are written in batches of 50 as `INSERT BULK` statements.

> **Scope of a dump:** The generated script preserves collection topology (dense vs hybrid) and all point payloads. It does **not** preserve quantization config, pinned model / vector dimensions, or payload indexes — those must be re-applied manually after import if needed.

**CLI usage:**
```bash
qql dump <collection_name> <output.qql>

# Override the default 50 points/INSERT BULK batch
qql dump <collection_name> <output.qql> --batch-size 200
```

**In-shell usage (inside the QQL REPL):**
```
qql> DUMP COLLECTION <name> <output.qql>
qql> DUMP <name> <output.qql>
```

Both forms are equivalent. The shorter `DUMP <name> <file>` form is a convenience shorthand.

**Example:**
```bash
qql dump medical_records /tmp/medical_records.qql
```

```
Dumping: 'medical_records'  →  /tmp/medical_records.qql

  Collection type : hybrid (dense + sparse)
  Points          : 41
  Batches         : 1  (50 points/batch)

  [1/1] wrote 41 point(s)

Done. 41 point(s) written.
```

**Generated file structure:**
```sql
-- ============================================================
-- QQL Dump — collection: medical_records
-- Generated : 2026-04-19 14:32:11
-- Points    : 41
-- Type      : hybrid (dense + sparse)
-- Note      : Re-importing re-embeds all text using the
--             configured model (see: qql connect).
-- ============================================================

CREATE COLLECTION medical_records HYBRID

-- Batch 1 / 1  (records 1–41)
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'Alzheimers disease is characterized by...',
    'title': 'Alzheimers Disease Overview',
    'department': 'neurology',
    'year': 2023,
    'peer_reviewed': true
  },
  ...
] USING HYBRID

-- ============================================================
-- End of dump
-- Written : 41
-- Skipped : 0  (no 'text' field)
-- ============================================================
```

**Round-trip workflow — data migration / partial restore:**
```bash
# 1. Dump the collection
qql dump medical_records backup.qql

# 2. Drop it
qql> DROP COLLECTION medical_records

# 3. Restore from the dump
qql execute backup.qql
```

**Rules and notes:**
- Points without a `'text'` payload field are **skipped** (counted in the footer comment).
- Hybrid collections produce `CREATE COLLECTION <name> HYBRID` and `INSERT BULK ... USING HYBRID` statements.
- Dense collections produce plain `CREATE COLLECTION <name>` and `INSERT BULK` statements.
- All payload value types are preserved: strings, integers, floats, booleans (`true`/`false`), `null`, lists, and nested dicts.
- Re-importing re-embeds all text using your currently configured model — use the same model as the original collection to preserve semantic accuracy.
- Parent directories of the output path are created automatically.
