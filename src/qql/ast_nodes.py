from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


class QuantizationType(Enum):
    SCALAR  = "scalar"
    BINARY  = "binary"
    PRODUCT = "product"
    TURBO   = "turbo"


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization settings parsed from a QUANTIZE clause."""
    type: QuantizationType
    quantile: float | None = None    # SCALAR only; None → Qdrant default (0.99)
    always_ram: bool = False         # all types; default False
    turbo_bits: float | None = None  # TURBO only; None → bits4 (Qdrant default 4-bit, 8×)


@dataclass(frozen=True)
class SearchWith:
    """Query-time search params supported by Qdrant SearchParams."""
    hnsw_ef: int | None = None
    exact: bool | None = None
    acorn: bool | None = None
    indexed_only: bool | None = None
    quantization: "QuantizationSearchWith | None" = None
    mmr_diversity: float | None = None
    mmr_candidates: int | None = None


@dataclass(frozen=True)
class QuantizationSearchWith:
    ignore: bool | None = None
    rescore: bool | None = None
    oversampling: float | None = None


@dataclass(frozen=True)
class VectorsConfig:
    on_disk: bool | None = None


@dataclass(frozen=True)
class HnswRuntimeConfig:
    m: int | None = None
    ef_construct: int | None = None
    full_scan_threshold: int | None = None
    max_indexing_threads: int | None = None
    on_disk: bool | None = None
    payload_m: int | None = None
    inline_storage: bool | None = None


@dataclass(frozen=True)
class OptimizersRuntimeConfig:
    deleted_threshold: float | None = None
    vacuum_min_vector_number: int | None = None
    default_segment_number: int | None = None
    max_segment_size: int | None = None
    memmap_threshold: int | None = None
    indexing_threshold: int | None = None
    flush_interval_sec: int | None = None
    max_optimization_threads: int | str | None = None
    prevent_unoptimized: bool | None = None


@dataclass(frozen=True)
class CollectionParamsConfig:
    replication_factor: int | None = None
    write_consistency_factor: int | None = None
    read_fan_out_factor: int | None = None
    read_fan_out_delay_ms: int | None = None
    on_disk_payload: bool | None = None


@dataclass(frozen=True)
class CollectionConfig:
    vectors: VectorsConfig | None = None
    hnsw: HnswRuntimeConfig | None = None
    optimizers: OptimizersRuntimeConfig | None = None
    params: CollectionParamsConfig | None = None


@dataclass(frozen=True)
class QuantizationUpdate:
    config: QuantizationConfig | None = None
    disabled: bool = False


# ── Filter expression leaf nodes ──────────────────────────────────────────────

@dataclass(frozen=True)
class CompareExpr:
    """field op literal  — covers =, !=, >, >=, <, <="""
    field: str
    op: str   # one of: "=", "!=", ">", ">=", "<", "<="
    value: str | int | float | bool


@dataclass(frozen=True)
class BetweenExpr:
    """field BETWEEN low AND high"""
    field: str
    low: int | float
    high: int | float


@dataclass(frozen=True)
class InExpr:
    """field IN (v1, v2, ...)"""
    field: str
    values: tuple[str | int | float | bool | None, ...]


@dataclass(frozen=True)
class NotInExpr:
    """field NOT IN (v1, v2, ...)"""
    field: str
    values: tuple[str | int | float | bool | None, ...]


@dataclass(frozen=True)
class IsNullExpr:
    """field IS NULL"""
    field: str


@dataclass(frozen=True)
class IsNotNullExpr:
    """field IS NOT NULL"""
    field: str


@dataclass(frozen=True)
class IsEmptyExpr:
    """field IS EMPTY"""
    field: str


@dataclass(frozen=True)
class IsNotEmptyExpr:
    """field IS NOT EMPTY"""
    field: str


@dataclass(frozen=True)
class MatchTextExpr:
    """field MATCH 'text'  — all terms required (MatchText)"""
    field: str
    text: str


@dataclass(frozen=True)
class MatchAnyExpr:
    """field MATCH ANY 'text'  — any term matches (MatchTextAny)"""
    field: str
    text: str


@dataclass(frozen=True)
class MatchPhraseExpr:
    """field MATCH PHRASE 'text'  — exact phrase (MatchPhrase)"""
    field: str
    text: str


# ── Filter expression logical nodes ──────────────────────────────────────────

@dataclass(frozen=True)
class AndExpr:
    """A AND B AND C — flattened into a single node with N operands."""
    operands: tuple[FilterExpr, ...]


@dataclass(frozen=True)
class OrExpr:
    """A OR B OR C"""
    operands: tuple[FilterExpr, ...]


@dataclass(frozen=True)
class NotExpr:
    """NOT A"""
    operand: FilterExpr


# Union type covering all filter expression nodes
FilterExpr = Union[
    CompareExpr, BetweenExpr, InExpr, NotInExpr,
    IsNullExpr, IsNotNullExpr, IsEmptyExpr, IsNotEmptyExpr,
    MatchTextExpr, MatchAnyExpr, MatchPhraseExpr,
    AndExpr, OrExpr, NotExpr,
]


# ── Statement nodes ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InsertStmt:
    collection: str
    values: dict[str, Any]  # must contain "text" key
    model: str | None       # dense model; None → use config default
    hybrid: bool = False            # if True, also embed + store sparse BM25 vector
    sparse_model: str | None = None # sparse model; None → SparseEmbedder.DEFAULT_MODEL
    dense_vector: str | None = None
    sparse_vector: str | None = None


@dataclass(frozen=True)
class InsertBulkStmt:
    collection: str
    values_list: tuple[dict[str, Any], ...]  # each dict must contain "text"
    model: str | None                         # dense model; None → use config default
    hybrid: bool = False
    sparse_model: str | None = None
    dense_vector: str | None = None
    sparse_vector: str | None = None


@dataclass(frozen=True)
class CreateCollectionStmt:
    collection: str
    hybrid: bool = False                      # if True, create with dense + sparse named vectors
    model: str | None = None                  # dense model; None → use config default
    quantization: QuantizationConfig | None = None  # optional QUANTIZE clause
    config: CollectionConfig | None = None
    dense_vector: str | None = None
    sparse_vector: str | None = None


@dataclass(frozen=True)
class AlterCollectionStmt:
    collection: str
    config: CollectionConfig | None = None
    quantization: QuantizationUpdate | None = None


@dataclass(frozen=True)
class CreateIndexStmt:
    collection: str
    field_name: str
    schema: str
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class DropCollectionStmt:
    collection: str


@dataclass(frozen=True)
class ShowCollectionsStmt:
    pass


@dataclass(frozen=True)
class ShowCollectionStmt:
    collection: str


@dataclass(frozen=True)
class SelectStmt:
    collection: str
    point_id: str | int


@dataclass(frozen=True)
class ScrollStmt:
    collection: str
    limit: int
    query_filter: FilterExpr | None = None
    after: str | int | None = None


@dataclass(frozen=True)
class SearchStmt:
    collection: str
    query_text: str
    limit: int
    model: str | None               # dense model; None → use config default
    hybrid: bool = False            # if True, use prefetch+RRF hybrid search
    fusion: str | None = None       # hybrid fusion strategy; None → default rrf
    sparse_only: bool = False       # if True, query only the sparse vector (no dense)
    sparse_model: str | None = None # sparse model for hybrid/sparse-only; None → SparseEmbedder.DEFAULT_MODEL
    query_filter: FilterExpr | None = None  # optional WHERE clause; default keeps existing tests valid
    rerank: bool = False                    # if True, apply cross-encoder reranking post-Qdrant
    rerank_model: str | None = None         # cross-encoder model; None → CrossEncoderEmbedder.DEFAULT_MODEL
    with_clause: SearchWith | None = None
    group_by: str | None = None             # GROUP BY field name; None → normal flat search
    group_size: int = 3                     # max points per group (ignored when group_by is None)
    dense_vector: str | None = None
    sparse_vector: str | None = None
    offset: int = 0                         # skip first N results
    score_threshold: float | None = None    # drop results below this score
    lookup_from: tuple[str, str | None] | None = None # cross-collection retrieval: (collection_name, vector_name)


@dataclass(frozen=True)
class RecommendStmt:
    collection: str
    positive_ids: tuple[str | int, ...]
    negative_ids: tuple[str | int, ...] = ()
    limit: int = 10
    strategy: str | None = None
    query_filter: FilterExpr | None = None
    offset: int = 0                         # skip first N results
    score_threshold: float | None = None    # drop results below this score
    with_clause: SearchWith | None = None
    lookup_from: tuple[str, str | None] | None = None # cross-collection retrieval: (collection_name, vector_name)
    using: str | None = None


@dataclass(frozen=True)
class DeleteStmt:
    collection: str
    point_id: str | int | None = None
    query_filter: FilterExpr | None = None


@dataclass(frozen=True)
class UpdateVectorStmt:
    """UPDATE <collection> SET VECTOR WHERE id = <id> [vector...]"""
    collection: str
    point_id: str | int
    vector: tuple[float, ...]   # dense vector as immutable tuple (frozen=True compatible)
    vector_name: str | None = None


@dataclass(frozen=True)
class UpdatePayloadStmt:
    """UPDATE <collection> SET PAYLOAD WHERE <filter|id> {payload}"""
    collection: str
    payload: dict[str, Any]
    point_id: str | int | None = None        # mutually exclusive with query_filter
    query_filter: FilterExpr | None = None


@dataclass(frozen=True)
class BatchBlockStmt:
    statements: tuple[ASTNode, ...]


# Union type for all top-level statement nodes
ASTNode = (
    InsertStmt
    | InsertBulkStmt
    | CreateCollectionStmt
    | AlterCollectionStmt
    | CreateIndexStmt
    | DropCollectionStmt
    | ShowCollectionsStmt
    | ShowCollectionStmt
    | SelectStmt
    | ScrollStmt
    | SearchStmt
    | RecommendStmt
    | DeleteStmt
    | UpdateVectorStmt
    | UpdatePayloadStmt
    | BatchBlockStmt
)
