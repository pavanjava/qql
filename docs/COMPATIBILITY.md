# QQL / Qdrant Compatibility Matrix

> Tracks the current Python `qql-cli` surface and known companion status for `qql-go`.
> Last checked: 2026-04-28.

This document should describe implemented behavior conservatively. If a feature is planned but not implemented, keep it marked as missing until tests exist.

## Legend

| Symbol | Meaning |
|---|---|
| Supported | Implemented and covered by normal usage/tests |
| Partial | Implemented with known limits |
| Missing | Not currently exposed by QQL |
| Planned | Roadmap or RFC candidate |
| Unknown | Needs verification in that implementation |

## Collection Management

| Feature | Qdrant API | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Create collection | `create_collection` | Supported | Supported | Dense collection |
| Create hybrid collection | `create_collection` with sparse vectors | Supported | Supported | `CREATE COLLECTION ... HYBRID` |
| Create with custom distance | Vector params | Missing | Missing | Currently cosine-only in Python |
| Create with custom HNSW | `hnsw_config` | Missing | Missing | Roadmap candidate |
| Create with quantization | `quantization_config` | Missing | Missing | Roadmap candidate |
| Create with on-disk payload | `on_disk_payload` | Missing | Missing | Roadmap candidate |
| Create with multivectors | `multivector_config` | Missing | Missing | Advanced roadmap candidate |
| Drop collection | `delete_collection` | Supported | Supported | `DROP COLLECTION` |
| List collections | `get_collections` | Supported | Supported | `SHOW COLLECTIONS` |
| Collection info | `get_collection` | Missing | Missing | Proposed as `DESCRIBE COLLECTION` |
| Collection aliases | Alias APIs | Missing | Missing | Later idea |
| Collection snapshots | Snapshot APIs | Missing | Missing | Later idea |

## Points / Documents

| Feature | Qdrant API | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Insert point | `upsert` | Supported | Supported | Requires a `text` field for embedding |
| Insert bulk | `upsert` | Supported | Supported | `INSERT BULK` |
| Explicit point ID on insert | `upsert` | Supported | Supported | Integer or UUID string |
| Get point by ID | `retrieve` | Missing | Missing | Near-term roadmap candidate |
| Update payload | `set_payload` | Missing | Missing | Near-term roadmap candidate |
| Delete point by ID | `delete` | Supported | Supported | `DELETE ... WHERE id = ...` |
| Delete points by filter | `delete` with filter selector | Supported | Supported | Python parser/executor support non-ID filters |
| Delete payload keys | `delete_payload` | Missing | Missing | Near-term roadmap candidate |
| Count points | `count` | Missing | Missing | Near-term roadmap candidate |
| Scroll points | `scroll` | Missing | Missing | Near-term roadmap candidate |

## Search

| Feature | Qdrant API | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Dense search | `query_points` | Supported | Supported | Default mode |
| Hybrid search | `query_points` + RRF | Supported | Supported | `USING HYBRID` |
| Sparse-only search | `query_points` sparse vector | Supported | Supported | `USING SPARSE` |
| Exact search | `SearchParams.exact` | Supported | Supported | `EXACT` or `WITH { exact: true }` |
| HNSW ef tuning | `SearchParams.hnsw_ef` | Supported | Supported | `WITH { hnsw_ef: N }` |
| ACORN filtered search | `SearchParams.acorn` | Supported | Supported | Depends on Qdrant support |
| Search with filters | `Filter` | Supported | Supported | `WHERE` clause |
| Search pagination | `offset` | Missing | Missing | Near-term roadmap candidate |
| Batch search | Batch/query APIs | Missing | Missing | Later idea |
| MMR diversity | Query diversity controls | Missing | Missing | Later idea |
| Score boosting | Formula/rescore APIs | Missing | Missing | Later idea |
| Multivector search | Multivector query | Missing | Missing | Later idea |
| Rerank | Cross-encoder / inference | Supported | Partial | Python uses local Fastembed cross-encoder; Go behavior should be checked against `qql-go` docs |
| Relevance feedback | Feedback query | Missing | Missing | Later idea |

## Recommend

| Feature | Qdrant API | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Recommend by examples | Recommend query | Supported | Supported | `RECOMMEND FROM` |
| Positive/negative IDs | Recommend input | Supported | Supported | |
| Strategy selection | `RecommendStrategy` | Supported | Supported | `average_vector`, `best_score`, `sum_scores` |
| Cross-collection lookup | `lookup_from` | Supported | Supported | |
| Named vector usage | `using` | Supported | Supported | |
| Offset | `offset` | Supported | Supported | |
| Score threshold | `score_threshold` | Supported | Supported | |
| Filtered recommend | `Filter` | Supported | Supported | `WHERE` clause |

## Payload Indexes

| Feature | Qdrant API | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Keyword index | `create_payload_index` | Supported | Supported | |
| Integer index | `create_payload_index` | Supported | Supported | Python syntax uses `TYPE integer` |
| Float index | `create_payload_index` | Supported | Supported | |
| Bool index | `create_payload_index` | Supported | Supported | |
| Text index | `create_payload_index` | Supported | Partial | Go support should be verified |
| Geo index | `create_payload_index` | Supported | Missing | Python maps `TYPE geo` |
| Datetime index | `create_payload_index` | Supported | Missing | Python maps `TYPE datetime` |

## Filtering

| Feature | Qdrant model | Python `qql-cli` | Go `qql-go` | Notes |
|---|---|---|---|---|
| Equality | `MatchValue` | Supported | Supported | `=` |
| Inequality | `must_not` + `MatchValue` | Supported | Supported | `!=` |
| Range | `Range` | Supported | Supported | `>`, `<`, `>=`, `<=` |
| Between | `Range` | Supported | Supported | Inclusive |
| In list | `MatchAny` | Supported | Supported | `IN (...)` |
| Not in list | `MatchExcept` | Supported | Supported | `NOT IN (...)` |
| Is null | `IsNull` | Supported | Supported | |
| Is empty | `IsEmpty` | Supported | Supported | |
| Full-text match | `MatchText` | Supported | Supported | `MATCH` |
| Match any term | `MatchTextAny` | Supported | Supported | `MATCH ANY` |
| Match phrase | `MatchPhrase` | Supported | Supported | `MATCH PHRASE` |
| Logical operators | `must`, `should`, `must_not` | Supported | Supported | `AND`, `OR`, `NOT` |
| Nested fields | Payload key paths | Supported | Supported | Dot notation |
| Nested array access | Payload key paths | Partial | Partial | Keep examples conservative until integration-tested |

## Version Notes

| Implementation | Current version in this repo/docs | Notes |
|---|---|---|
| Python `qql-cli` | `1.4.0` | Source of truth for this repository |
| Go `qql-go` | `0.1.x` | Companion implementation; verify exact behavior in the Go repo before release claims |

## Known Gaps

| Gap | Impact | Suggested next step |
|---|---|---|
| No `GET` statement | Hard to inspect one point from the CLI | Add RFC or issue |
| No `SCROLL` statement | Hard to page/export large collections through QQL syntax | Add RFC or issue |
| No `COUNT` statement | Hard to validate scripts and filters | Add RFC or issue |
| No `DESCRIBE COLLECTION` | Users must drop to SDK/Qdrant UI for collection metadata | Add RFC or issue |
| No payload update syntax | Metadata updates require SDK calls or full reinsert | Add RFC or issue |
| Limited custom collection configuration | Advanced users need SDK for distance/HNSW/quantization | Define minimal syntax before implementing |

## Maintenance Rule

When changing QQL behavior:

1. Update this matrix in the same PR.
2. Link or mention tests that prove the status.
3. Mark companion implementation status as `Unknown` rather than guessing.
4. Avoid future-tense claims unless there is an accepted RFC or linked issue.
