---
layout: default
title: "SEARCH / SELECT / RECOMMEND"
---

# SEARCH, SELECT, SCROLL, RECOMMEND, Hybrid Search & Reranking

---

## SEARCH — find similar points

Performs a **semantic similarity search**: your query text is embedded with the same model used during insert, then Qdrant finds the nearest vectors by cosine distance.

An optional `WHERE` clause filters the candidate set **before** similarity ranking so you only get results that match both the semantic query and the payload conditions.

**Syntax:**
```
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [OFFSET <n>] [SCORE THRESHOLD <f>] [LOOKUP FROM <collection> [VECTOR '<name>']]
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING MODEL '<model_name>'
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING VECTOR '<dense_vector_name>'
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING MODEL '<model>'] WHERE <filter>
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING HYBRID
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING HYBRID [FUSION 'rrf|dbsf'] [DENSE MODEL '<model>'] [DENSE VECTOR '<name>'] [SPARSE MODEL '<model>'] [SPARSE VECTOR '<name>'] [WHERE <filter>]
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING SPARSE [MODEL '<sparse_model>'] [VECTOR '<sparse_vector_name>']
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> EXACT
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING ...] [WHERE <filter>] [RERANK] WITH { hnsw_ef: <n>, exact: true|false, acorn: true|false, indexed_only: true|false, quantization: { ignore: true|false, rescore: true|false, oversampling: <n> }, mmr_diversity: <0..1>, mmr_candidates: <n> }
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING ...] [WHERE <filter>] RERANK [MODEL '<reranker_model>']
```

**Examples:**

Basic search, return top 5 results:
```sql
SEARCH articles SIMILAR TO 'machine learning algorithms' LIMIT 5
```

Pagination with OFFSET:
```sql
SEARCH articles SIMILAR TO 'machine learning' LIMIT 10 OFFSET 20
```

Filter low-quality matches with SCORE THRESHOLD:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 SCORE THRESHOLD 0.8
```

Cross-collection vector lookup:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 5 LOOKUP FROM user_profiles VECTOR 'preferences'
```

Search only papers published after 2020:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 WHERE year > 2020
```

Hybrid search (combines dense semantic + sparse BM25 keyword retrieval via RRF by default):
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 USING HYBRID
```

Search a specific named dense vector:
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 USING VECTOR 'body'
```

Hybrid search against external vector names:
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 USING HYBRID DENSE VECTOR 'emb' SPARSE VECTOR 'lex'
```

Sparse-only search (queries a sparse vector — useful for pure keyword retrieval):
```sql
SEARCH medical_knowledge SIMILAR TO 'beta blocker contraindications' LIMIT 5 USING SPARSE
```

> **Sparse scores are unbounded dot-products.** Unlike dense cosine similarity (which is bounded 0–1), sparse vector scores are raw dot-products that can exceed 1.0 — scores like 8.3 or 14.5 are perfectly normal and expected. Do not compare sparse scores to dense cosine scores; they are on different scales.

Exact search for recall debugging:
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 EXACT
```

Search with query-time HNSW tuning:
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 WITH { hnsw_ef: 128 }
```

Search with native MMR diversification:
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 WITH { mmr_diversity: 0.5, mmr_candidates: 50 }
```

**Clause Order:**
`SEARCH` requires clauses to appear in this strict order if used:
`LIMIT` → `OFFSET` → `SCORE THRESHOLD` → `LOOKUP FROM` → `USING ...` → `WHERE` → `RERANK` → `WITH` → `GROUP BY`

**Output:**

Results are displayed as a table with three columns:

```
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼──────────────────────────────────
 0.9241 │ 3f2e1a4b-...                          │ {'text': 'Neural networks...', 'author': 'alice'}
 0.8817 │ 7a1b2c3d-...                          │ {'text': 'Attention is all...', 'tags': [...]}
```

**Important:** Use the same model for SEARCH as you used for INSERT. Mixing models produces meaningless scores because the vectors live in different spaces.

---

## SELECT — retrieve a point by ID

Fetches a single point payload by exact point ID.

**Syntax:**
```sql
SELECT * FROM <collection_name> WHERE id = '<point_id>'
SELECT * FROM <collection_name> WHERE id = <integer_id>
```

**Examples:**
```sql
SELECT * FROM articles WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456'
SELECT * FROM articles WHERE id = 42
```

`SELECT` in this version is intentionally strict:
- only `*` projection is supported
- only `WHERE id = ...` is supported

---

## Query-Time Search Params (`EXACT`, `WITH`)

Use these when you want to debug retrieval quality or tune recall without changing collection-level settings.

| Syntax | Effect |
|---|---|
| `EXACT` | Shorthand for exact KNN search (`exact=true`) |
| `WITH { hnsw_ef: 128 }` | Increase HNSW exploration at query time |
| `WITH { exact: true }` | Force exact KNN explicitly |
| `WITH { acorn: true }` | Enable ACORN for filtered queries |
| `WITH { indexed_only: true, quantization: { rescore: true } }` | Prefer indexed vectors and apply quantization controls |
| `WITH { mmr_diversity: 0.5, mmr_candidates: 50 }` | Apply native MMR diversification after nearest-neighbor retrieval |

- `EXACT` can appear after `LIMIT` or after `RERANK`
- `WITH { ... }` can appear after `WHERE` and/or `RERANK`
- Supported top-level `WITH` keys are `hnsw_ef`, `exact`, `acorn`, `indexed_only`, `quantization`, `mmr_diversity`, and `mmr_candidates`
- MMR is currently supported for dense `SEARCH` and dense `SEARCH ... GROUP BY`
- MMR is not yet supported with `USING HYBRID`, `USING SPARSE`, or `RECOMMEND`

```sql
-- Exact KNN baseline
SEARCH articles SIMILAR TO 'programming language' LIMIT 5 EXACT

-- Raise HNSW ef at query time
SEARCH articles SIMILAR TO 'transformers' LIMIT 10 WITH { hnsw_ef: 256 }

-- Filtered search with ACORN
SEARCH articles SIMILAR TO 'RAG' LIMIT 10 WHERE tag = 'li' WITH { acorn: true }

-- Restrict to indexed segments only
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WITH { indexed_only: true }

-- Quantized-search tuning
SEARCH articles SIMILAR TO 'vector db' LIMIT 10 WITH { quantization: { ignore: true, oversampling: 2 } }

-- Diversify top-k results with native MMR
SEARCH articles SIMILAR TO 'retrieval systems' LIMIT 10 WITH { mmr_diversity: 0.5, mmr_candidates: 50 }
```

---

## SCROLL — pagination / browsing

Use `SCROLL` to iterate through points in a collection page by page.

**Syntax:**
```sql
SCROLL FROM <collection_name> LIMIT <n>
SCROLL FROM <collection_name> WHERE <filter> LIMIT <n>
SCROLL FROM <collection_name> AFTER '<point_id>' LIMIT <n>
SCROLL FROM <collection_name> WHERE <filter> AFTER <point_id> LIMIT <n>
```

**Examples:**
```sql
SCROLL FROM articles LIMIT 50
SCROLL FROM articles WHERE year >= 2024 LIMIT 50
SCROLL FROM articles AFTER 'cursor-id' LIMIT 50
```

**Behavior:**
- Returns points in ID order with payloads.
- Returns a `next_offset` cursor when more points are available.
- `next_offset` preserves the native point-id type (`string` or integer).
- Use `AFTER <next_offset>` to fetch the next page.

---

## Hybrid Search (USING HYBRID)

Hybrid search combines **dense semantic vectors** and **sparse BM25 keyword vectors** in a single query. By default QQL merges the two result sets with Qdrant's **Reciprocal Rank Fusion (RRF)** algorithm, and you can optionally switch to **DBSF** with a `FUSION` clause.

### How it works internally

1. Both a dense vector (`TextEmbedding`) and a sparse BM25 vector (`SparseTextEmbedding`) are generated from your query text.
2. Qdrant fetches the top candidates from each index independently (`prefetch limit = LIMIT × 4`).
3. The two result lists are merged using the selected fusion strategy (`RRF` by default, or `DBSF` when requested).
4. The final top-N results are returned.

### Step 1: Create a hybrid collection

```sql
-- Shorthand (backward compatible)
CREATE COLLECTION articles HYBRID

-- USING form — allows specifying a dense model
CREATE COLLECTION articles USING HYBRID
CREATE COLLECTION articles USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
```

### Step 2: Insert with hybrid vectors

```sql
INSERT INTO COLLECTION articles VALUES {
  'text': 'Attention is all you need',
  'author': 'Vaswani et al.',
  'year': 2017
} USING HYBRID
```

### Step 3: Search with hybrid retrieval

```sql
-- Basic hybrid search
SEARCH articles SIMILAR TO 'transformer architecture' LIMIT 10 USING HYBRID

-- Hybrid search with a WHERE filter
SEARCH articles SIMILAR TO 'attention' LIMIT 10 USING HYBRID WHERE year >= 2017

-- Hybrid with DBSF fusion
SEARCH articles SIMILAR TO 'hybrid retrieval' LIMIT 10 USING HYBRID FUSION 'dbsf'

-- Hybrid with custom dense model
SEARCH articles SIMILAR TO 'embeddings' LIMIT 5
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'

-- Hybrid with both custom models
SEARCH articles SIMILAR TO 'sparse retrieval' LIMIT 5
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'prithivida/Splade_PP_en_v1'
```

### Model defaults in hybrid mode

| Argument | Default |
|---|---|
| Dense model | configured default (`sentence-transformers/all-MiniLM-L6-v2`) |
| Sparse model | `Qdrant/bm25` |
| Fusion | `rrf` |

### Dense vs. hybrid — when to use which

| Situation | Recommendation |
|---|---|
| Semantic similarity (paraphrasing, synonyms) | Dense only |
| Exact keyword matching (product codes, names) | Hybrid or BM25-only |
| General-purpose retrieval (unknown query distribution) | Hybrid |
| Low latency / small collection | Dense only |

---

## RECOMMEND — retrieve by example IDs

Performs a Qdrant recommendation query using existing point IDs as positive and optional negative examples. Qdrant uses those examples to retrieve nearby points, and QQL automatically excludes the seed IDs from the results.

**Syntax:**
```sql
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) NEGATIVE IDS (<id>, ...) LIMIT <n>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) STRATEGY '<strategy>' LIMIT <n>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> WHERE <filter>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> OFFSET <n>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> SCORE THRESHOLD <f>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> WITH { exact: true, hnsw_ef: <n>, indexed_only: true|false, quantization: { ignore: true|false, rescore: true|false, oversampling: <n> } }
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> LOOKUP FROM <collection>
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> LOOKUP FROM <collection> VECTOR '<name>'
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> USING '<vector_name>'
```

**Examples:**

Recommend more results like two known articles:
```sql
RECOMMEND FROM articles POSITIVE IDS (1001, 1002) LIMIT 5
```

Recommend similar results while steering away from one bad example:
```sql
RECOMMEND FROM articles POSITIVE IDS (1001, 1002) NEGATIVE IDS (1009) LIMIT 5
```

Use Qdrant's `best_score` recommendation strategy:
```sql
RECOMMEND FROM articles POSITIVE IDS (1001) STRATEGY 'best_score' LIMIT 10
```

Recommend only within a filtered subset:
```sql
RECOMMEND FROM articles POSITIVE IDS (1001) LIMIT 5 WHERE year >= 2020 AND status = 'published'
```

Cross-collection recommend (look up example IDs from another collection):
```sql
RECOMMEND FROM target_collection
  POSITIVE IDS ('a')
  LOOKUP FROM source_collection VECTOR 'dense'
  LIMIT 5
```

Full-featured recommend:
```sql
RECOMMEND FROM articles
  POSITIVE IDS (1001, 1002)
  NEGATIVE IDS (1009)
  STRATEGY 'best_score'
  LOOKUP FROM other_collection VECTOR 'dense'
  USING 'dense'
  LIMIT 10
  OFFSET 5
  SCORE THRESHOLD 0.5
  WHERE year >= 2020
  WITH { exact: true }
```

**Supported strategies:** `average_vector`, `best_score`, `sum_scores`

**Clause order:** `POSITIVE IDS` → `NEGATIVE IDS` → `STRATEGY` → `LOOKUP FROM` → `USING` → `LIMIT` → `OFFSET` → `SCORE THRESHOLD` → `WHERE` → `WITH`

---

## Cross-Encoder Reranking (RERANK)

Appending `RERANK` to any SEARCH statement activates a **second-pass relevance scoring** step using a cross-encoder model. Cross-encoders process the **(query, document)** pair jointly, producing a more accurate relevance score at the cost of extra compute.

### How it works internally

1. Qdrant executes the normal dense or hybrid search, but fetches `LIMIT × 4` candidates.
2. Each candidate's `payload["text"]` is paired with the original query text.
3. The cross-encoder scores all (query, document) pairs in one batch.
4. Results are sorted **descending by cross-encoder score** and sliced to `LIMIT`.
5. The `score` column reflects the cross-encoder relevance score (raw logits — higher is more relevant).

**Syntax:**
```
SEARCH <name> SIMILAR TO '<query>' LIMIT <n> RERANK
SEARCH <name> SIMILAR TO '<query>' LIMIT <n> RERANK MODEL '<cross_encoder_model>'
SEARCH ... LIMIT n [USING ...] [WHERE ...] RERANK [MODEL '...']
```

**Examples:**

Dense search + rerank (default cross-encoder):
```sql
SEARCH articles SIMILAR TO 'machine learning for healthcare' LIMIT 5 RERANK
```

Hybrid search + rerank (best of all three worlds):
```sql
SEARCH articles SIMILAR TO 'attention mechanism in transformers' LIMIT 10 USING HYBRID RERANK
```

Custom cross-encoder model:
```sql
SEARCH articles SIMILAR TO 'semantic search' LIMIT 5
  RERANK MODEL 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

**Default cross-encoder model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Model | Notes |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Default. Fast and accurate for passage reranking |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Larger, higher quality, slower |
| `BAAI/bge-reranker-base` | BGE reranker, strong general-purpose performance |
| `BAAI/bge-reranker-large` | Highest quality BGE reranker, slower |

### When to use RERANK

| Situation | Recommendation |
|---|---|
| High-precision retrieval (legal, medical, research) | Add `RERANK` |
| Small LIMIT (top-3 or top-5 results) | Very effective |
| Low latency required | Skip `RERANK` (adds ~100–500 ms per batch) |
| Large collections with keyword-heavy queries | `USING HYBRID RERANK` |

> **Note on scores:** After reranking, the `score` column shows the cross-encoder's raw logit (can be any real number, unbounded). Do not compare reranked scores to non-reranked cosine similarity scores.

---

## SEARCH … GROUP BY — grouped results

Returns the top-scoring points **grouped by a payload field value**. Instead of a single flat ranked list, results are organised into groups — each group contains the top-scoring points that share the same value for the specified field.

Useful for **result diversification**: e.g. "return the 3 best articles from each category", or "show the top 2 papers per author".

**Syntax:**
```
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> GROUP BY <field>
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> GROUP BY <field> GROUP_SIZE <m>
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> [WHERE <filter>] GROUP BY <field> [GROUP_SIZE <m>]
SEARCH <collection> SIMILAR TO '<query>' LIMIT <n> USING HYBRID GROUP BY <field> [GROUP_SIZE <m>]
```

- **`LIMIT <n>`** — maximum number of **groups** to return.
- **`GROUP_SIZE <m>`** — maximum number of points per group (default: **3**).
- **`GROUP BY <field>`** — the payload field whose values define the groups. **Must be a string (keyword) or number (integer) field** — this is enforced by Qdrant. Dot-notation is supported (e.g. `meta.author`). Array-valued fields are allowed: a point with multiple values for the field can appear in multiple groups. The field should be indexed as `keyword` or `integer` for best performance (see [CREATE INDEX](collections.md)).
- `WHERE` filters, `USING HYBRID`, and `USING MODEL` are all compatible with GROUP BY.
- ⚠️ **Incompatibility:** `GROUP BY` is not compatible with `OFFSET` or `RERANK`. Use cursors (not currently supported in QQL) for paginating grouped results in Qdrant.
- **`GROUP BY` and `RERANK` cannot be combined** in the same statement — this raises a syntax error.

**Examples:**

Top 5 categories, up to 3 articles each (default group_size):
```sql
SEARCH articles SIMILAR TO 'machine learning' LIMIT 5 GROUP BY category
```

Top 3 authors, up to 2 papers each:
```sql
SEARCH papers SIMILAR TO 'neural networks' LIMIT 3 GROUP BY author GROUP_SIZE 2
```

Grouped search with a payload filter:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 5 WHERE year >= 2022 GROUP BY category GROUP_SIZE 4
```

Grouped hybrid search:
```sql
SEARCH articles SIMILAR TO 'vector databases' LIMIT 4 USING HYBRID GROUP BY category GROUP_SIZE 3
```

**Output:**

```
✓ Found 3 group(s) by 'category' (grouped)
Group: machine-learning
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼────────────────────────────────────────
 0.9312 │ 3f2e1a4b-8c91-4d0e-b123-abc123def456 │ {'text': '...', 'category': 'machine-learning'}
 0.8845 │ 9a1b2c3d-4e5f-6789-abcd-ef0123456789 │ {'text': '...', 'category': 'machine-learning'}

Group: nlp
 Score  │ ID                                   │ Payload
────────┼──────────────────────────────────────┼────────────────────────────────────────
 0.9100 │ 1a2b3c4d-5e6f-7890-bcde-f01234567890 │ {'text': '...', 'category': 'nlp'}
```

> **Tip:** For GROUP BY to work efficiently, create a payload index on the grouping field first:
> ```sql
> CREATE INDEX ON COLLECTION articles FOR category TYPE keyword
> ```
