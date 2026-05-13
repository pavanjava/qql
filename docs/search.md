# SEARCH, SCROLL, RECOMMEND, Hybrid Search & Reranking

---

## SEARCH — find similar points

Performs a **semantic similarity search**: your query text is embedded with the same model used during insert, then Qdrant finds the nearest vectors by cosine distance.

An optional `WHERE` clause filters the candidate set **before** similarity ranking so you only get results that match both the semantic query and the payload conditions.

**Syntax:**
```
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n>
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING MODEL '<model_name>'
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING MODEL '<model>'] WHERE <filter>
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING HYBRID
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING HYBRID [DENSE MODEL '<model>'] [SPARSE MODEL '<model>'] [WHERE <filter>]
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> USING SPARSE [MODEL '<sparse_model>']
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> EXACT
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING ...] [WHERE <filter>] [RERANK] WITH { hnsw_ef: <n>, exact: true|false, acorn: true|false }
SEARCH <collection_name> SIMILAR TO '<query_text>' LIMIT <n> [USING ...] [WHERE <filter>] RERANK [MODEL '<reranker_model>']
```

**Examples:**

Basic search, return top 5 results:
```sql
SEARCH articles SIMILAR TO 'machine learning algorithms' LIMIT 5
```

Search only papers published after 2020:
```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 WHERE year > 2020
```

Hybrid search (combines dense semantic + sparse BM25 keyword retrieval via RRF):
```sql
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 USING HYBRID
```

Sparse-only search (queries only the `sparse` named vector — useful for pure keyword retrieval):
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

## Query-Time Search Params (`EXACT`, `WITH`)

Use these when you want to debug retrieval quality or tune recall without changing collection-level settings.

| Syntax | Effect |
|---|---|
| `EXACT` | Shorthand for exact KNN search (`exact=true`) |
| `WITH { hnsw_ef: 128 }` | Increase HNSW exploration at query time |
| `WITH { exact: true }` | Force exact KNN explicitly |
| `WITH { acorn: true }` | Enable ACORN for filtered queries |

- `EXACT` can appear after `LIMIT` or after `RERANK`
- `WITH { ... }` can appear after `WHERE` and/or `RERANK`
- Supported `WITH` keys are only `hnsw_ef`, `exact`, and `acorn`

```sql
-- Exact KNN baseline
SEARCH articles SIMILAR TO 'programming language' LIMIT 5 EXACT

-- Raise HNSW ef at query time
SEARCH articles SIMILAR TO 'transformers' LIMIT 10 WITH { hnsw_ef: 256 }

-- Filtered search with ACORN
SEARCH articles SIMILAR TO 'RAG' LIMIT 10 WHERE tag = 'li' WITH { acorn: true }
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
- Use `AFTER <next_offset>` to fetch the next page.

---

## Hybrid Search (USING HYBRID)

Hybrid search combines **dense semantic vectors** and **sparse BM25 keyword vectors** in a single query and merges the results with Qdrant's **Reciprocal Rank Fusion (RRF)** algorithm. This typically outperforms either method alone.

### How it works internally

1. Both a dense vector (`TextEmbedding`) and a sparse BM25 vector (`SparseTextEmbedding`) are generated from your query text.
2. Qdrant fetches the top candidates from each index independently (`prefetch limit = LIMIT × 4`).
3. The two result lists are merged using RRF — a rank-based fusion that does not require score normalization.
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
RECOMMEND FROM <collection_name> POSITIVE IDS (<id>, ...) LIMIT <n> WITH { exact: true, hnsw_ef: <n> }
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
