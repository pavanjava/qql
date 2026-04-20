### Concept: Create Collection
```commandline
-- Dense-only (default model, 384 dims)
CREATE COLLECTION research_papers

-- Pinned to a specific model (768 dims)
CREATE COLLECTION research_papers USING MODEL 'BAAI/bge-base-en-v1.5'

-- Hybrid (dense + sparse BM25)
CREATE COLLECTION research_papers HYBRID

-- Hybrid with custom dense model
CREATE COLLECTION research_papers USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
```

### Concept: SHOW COLLECTIONS + DROP COLLECTION
```commandline
SHOW COLLECTIONS

DROP COLLECTION old_experiments
```

### Concept: INSERT INTO COLLECTION
```commandline
-- Minimal (text only)
INSERT INTO COLLECTION articles VALUES {'text': 'Qdrant supports cosine similarity search'}

-- With rich metadata
INSERT INTO COLLECTION articles VALUES {
  'text': 'Neural networks learn representations from data',
  'author': 'alice',
  'category': 'ml',
  'year': 2024,
  'published': true
}

-- With a specific embedding model
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'} USING MODEL 'BAAI/bge-small-en-v1.5'
```
#### Bulk Insert
```commandline
-- Minimal bulk
INSERT BULK INTO COLLECTION articles VALUES [
  {'text': 'Qdrant supports cosine similarity search'},
  {'text': 'Sparse BM25 vectors enable keyword retrieval'},
  {'text': 'Hybrid search combines dense and sparse results via RRF'}
]

-- With metadata
INSERT BULK INTO COLLECTION articles VALUES [
  {'text': 'Attention is all you need', 'author': 'vaswani', 'year': 2017},
  {'text': 'BERT: Pre-training of deep bidirectional transformers', 'author': 'devlin', 'year': 2018},
  {'text': 'Language models are few-shot learners', 'author': 'brown', 'year': 2020}
]
```

### Concept: DELETE FROM
```commandline
-- By UUID (from INSERT output)
DELETE FROM articles WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456'

-- By integer ID
DELETE FROM articles WHERE id = 42
```

### Concept: Basic Semantic Search
```commandline
-- Top 5 results
SEARCH articles SIMILAR TO 'machine learning algorithms' LIMIT 5

-- With a specific model (must match INSERT model)
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 USING MODEL 'BAAI/bge-small-en-v1.5'

-- Equality / inequality
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE category = 'paper'
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE status != 'draft'

-- Range
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE year > 2020
SEARCH articles SIMILAR TO 'history of ai' LIMIT 10 WHERE year BETWEEN 2018 AND 2023

-- IN / NOT IN
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE status IN ('published', 'reviewed')

-- Null checks
SEARCH articles SIMILAR TO 'ml' LIMIT 5 WHERE reviewer IS NOT NULL

-- Full-text MATCH
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH PHRASE 'semantic search'

-- Logical AND / OR / NOT
SEARCH articles SIMILAR TO 'nlp' LIMIT 10 WHERE category = 'paper' AND year >= 2020
SEARCH articles SIMILAR TO 'conference' LIMIT 10 WHERE (source = 'arxiv' OR source = 'ieee') AND year >= 2022

-- Single level nesting
SEARCH articles SIMILAR TO 'wikipedia' LIMIT 5 WHERE meta.source = 'web'

-- Array of nested objects
SEARCH cities SIMILAR TO 'large city' LIMIT 5 WHERE country.cities[].population > 1000000

-- Simple Hybrid search
SEARCH articles SIMILAR TO 'transformer architecture' LIMIT 10 USING HYBRID

-- Hybrid search with WHERE
SEARCH articles SIMILAR TO 'transformers' LIMIT 10 USING HYBRID WHERE year >= 2020

-- Custom dense + sparse models
SEARCH articles SIMILAR TO 'sparse retrieval' LIMIT 5
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'prithivida/Splade_PP_en_v1'
  
-- Default SPLADE/BM25
SEARCH medical_knowledge SIMILAR TO 'beta blocker contraindications' LIMIT 5 USING SPARSE

-- Custom sparse model
SEARCH medical_knowledge SIMILAR TO 'beta blocker contraindications' LIMIT 5
  USING SPARSE MODEL 'prithivida/Splade_PP_en_v1'
```

### Concept: RERANK — second-pass precision scoring
```commandline
-- Dense search + rerank
SEARCH articles SIMILAR TO 'machine learning for healthcare' LIMIT 5 RERANK

-- Hybrid + rerank (maximum precision)
SEARCH articles SIMILAR TO 'attention mechanism in transformers' LIMIT 10 USING HYBRID RERANK

-- With WHERE + rerank
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10 WHERE year > 2020 RERANK

-- Custom cross-encoder
SEARCH articles SIMILAR TO 'semantic search' LIMIT 5
  RERANK MODEL 'BAAI/bge-reranker-large'

-- Everything combined
SEARCH articles SIMILAR TO 'neural IR' LIMIT 10
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
  WHERE year >= 2020
  RERANK MODEL 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

### Concept: Query-time search parameter overrides
```commandline
-- Exact KNN (brute force, useful for recall debugging)
SEARCH articles SIMILAR TO 'attention mechanism' LIMIT 10 EXACT

-- Boost HNSW exploration at query time (higher ef = better recall, slower)
SEARCH articles SIMILAR TO 'transformers' LIMIT 10 WITH { hnsw_ef: 256 }

-- ACORN for filtered queries
SEARCH articles SIMILAR TO 'RAG' LIMIT 10 WHERE tag = 'li' WITH { acorn: true }

-- Hybrid + exact mode
SEARCH articles SIMILAR TO 'attention' LIMIT 10 USING HYBRID EXACT
```

### Concept: playing with .qql script files
```commandline
# From terminal
qql> execute seed.qql

# Stop on first error
qql> execute seed.qql --stop-on-error

# Export all points to a .qql file
qql> dump medical_records backup.qql

# Inside the shell
qql> DUMP COLLECTION medical_records backup.qql

# Round-trip: backup → drop → restore
qql> dump medical_records backup.qql
qql> DROP COLLECTION medical_records
qql> execute backup.qql
```