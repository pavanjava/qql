---
layout: default
title: "INSERT / INSERT BULK"
---

# INSERT — Adding Documents to Qdrant

QQL provides two insert statements: `INSERT` for single documents and `INSERT BULK` for batch inserts.

---

## INSERT — add a point

Inserts a new document into a collection. The `text` field is **mandatory** — it is automatically vectorized and stored as the point's vector. All other fields become searchable payload (metadata).

If the collection does not exist yet, it is **created automatically** with the correct vector dimensions.

If you include an `id` field in `VALUES`, QQL uses it as the Qdrant point ID. Supported explicit IDs are unsigned integers or UUID strings. If you omit `id`, QQL generates a UUID automatically.

**Syntax:**
```
INSERT INTO COLLECTION <collection_name> VALUES {<dict>}
INSERT INTO COLLECTION <collection_name> VALUES {<dict>} USING MODEL '<model_name>'
INSERT INTO COLLECTION <collection_name> VALUES {<dict>} USING VECTOR '<dense_vector_name>'
INSERT INTO COLLECTION <collection_name> VALUES {<dict>} USING HYBRID
INSERT INTO COLLECTION <collection_name> VALUES {<dict>} USING HYBRID [DENSE MODEL '<model>'] [DENSE VECTOR '<name>'] [SPARSE MODEL '<model>'] [SPARSE VECTOR '<name>']
```

**Examples:**

Minimal insert (text only):
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'Qdrant supports cosine similarity search'}
```

Insert with metadata:
```sql
INSERT INTO COLLECTION articles VALUES {
  'id': 1001,
  'text': 'Neural networks learn representations from data',
  'author': 'alice',
  'category': 'ml',
  'year': 2024,
  'published': true
}
```

Insert with a specific embedding model:
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'} USING MODEL 'BAAI/bge-small-en-v1.5'
```

Insert into a hybrid collection (dense + sparse BM25 vectors):
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'Attention is all you need'} USING HYBRID
```

Insert into a specific named dense vector:
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'} USING VECTOR 'body'
```

Insert into a hybrid collection with external vector names:
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'}
  USING HYBRID DENSE VECTOR 'emb' SPARSE VECTOR 'lex'
```

Insert with custom models for both dense and sparse:
```sql
INSERT INTO COLLECTION articles VALUES {'text': 'hello world'}
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'prithivida/Splade_PP_en_v1'
```

**What happens internally:**
1. The `text` value is embedded into a dense vector using the configured model.
2. In hybrid mode, a sparse BM25 vector is also generated.
3. If `id` is provided, it is used as the point ID; otherwise a UUID is auto-generated.
4. All fields except `id` are stored in the payload.
5. The point is upserted into Qdrant.

**Rules:**
- `text` is always required. Omitting it raises an error.
- `id`, when provided, must be an unsigned integer or UUID string.
- If the collection already exists with a different vector size (from a different model), an error is raised with a clear message.
- Hybrid inserts require a hybrid collection (created with `CREATE COLLECTION ... HYBRID`, auto-created on the first `USING HYBRID` insert, or **auto-detected** — if you omit `USING HYBRID` but the target collection is already a hybrid collection, QQL detects this and uses the hybrid insert path automatically).
- If a collection has multiple dense or sparse vectors, specify the target vector names explicitly.

---

## INSERT BULK — batch insert multiple points

Inserts multiple documents in a single statement. Each item in the array must contain a `"text"` key. All items are embedded and upserted to Qdrant in **one batched call**, which is significantly faster than issuing one `INSERT` per record.

If the collection does not exist yet, it is **created automatically** on the first bulk insert.

Each record may optionally include an `id` field. This is the preferred way to keep seed data deterministic and to make follow-up operations like `RECOMMEND` or `DELETE` reproducible.

**Syntax:**
```
INSERT BULK INTO COLLECTION <collection_name> VALUES [<dict>, <dict>, ...]
INSERT BULK INTO COLLECTION <collection_name> VALUES [<dict>, ...] USING MODEL '<model_name>'
INSERT BULK INTO COLLECTION <collection_name> VALUES [<dict>, ...] USING VECTOR '<dense_vector_name>'
INSERT BULK INTO COLLECTION <collection_name> VALUES [<dict>, ...] USING HYBRID
INSERT BULK INTO COLLECTION <collection_name> VALUES [<dict>, ...] USING HYBRID [DENSE MODEL '<model>'] [DENSE VECTOR '<name>'] [SPARSE MODEL '<model>'] [SPARSE VECTOR '<name>']
```

**Examples:**

Minimal bulk insert (text only):
```sql
INSERT BULK INTO COLLECTION articles VALUES [
  {'text': 'Qdrant supports cosine similarity search'},
  {'text': 'Sparse BM25 vectors enable keyword retrieval'},
  {'text': 'Hybrid search combines dense and sparse results via RRF'}
]
```

Bulk insert with metadata:
```sql
INSERT BULK INTO COLLECTION articles VALUES [
  {'id': 1001, 'text': 'Attention is all you need', 'author': 'vaswani', 'year': 2017},
  {'id': 1002, 'text': 'BERT: Pre-training of deep bidirectional transformers', 'author': 'devlin', 'year': 2018},
  {'id': 1003, 'text': 'Language models are few-shot learners', 'author': 'brown', 'year': 2020}
]
```

Bulk insert into a hybrid collection:
```sql
INSERT BULK INTO COLLECTION articles VALUES [
  {'text': 'Dense retrieval with FAISS', 'domain': 'ir'},
  {'text': 'Sparse retrieval with BM25', 'domain': 'ir'}
] USING HYBRID
```

**Rules:**
- Every dict in the array must contain a `"text"` key. Missing `text` on any item raises an error with the offending index.
- An empty array `[]` raises an error.
- `id`, when provided, must be an unsigned integer or UUID string.
- Supports all the same `USING` clauses as single `INSERT`.

---

## Value Types in Payload Dicts

The `VALUES` dictionary (and nested dicts) supports these types:

| Type | Example | Notes |
|---|---|---|
| String | `'hello'` or `"hello"` | Single or double quotes |
| Integer | `42`, `-7` | Whole numbers, negative allowed |
| Float | `3.14`, `-0.5` | Decimal numbers |
| Boolean | `true`, `false` | Case-insensitive |
| Null | `null` | Case-insensitive |
| Nested dict | `{'key': 'val'}` | Arbitrary nesting |
| List | `['a', 'b', 1]` | Mixed types allowed |

**Example using every type:**
```sql
INSERT INTO demo VALUES {
  'text':    'example document',
  'count':   42,
  'score':   0.95,
  'active':  true,
  'deleted': false,
  'ref':     null,
  'meta':    {'source': 'web', 'lang': 'en'},
  'tags':    ['ai', 'nlp', 'search']
}
```

Trailing commas in dicts and lists are allowed:
```sql
INSERT INTO demo VALUES {'text': 'hi', 'x': 1,}
```
