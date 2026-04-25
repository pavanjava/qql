from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class SearchWith:
    """Query-time search params supported by Qdrant SearchParams."""
    hnsw_ef: int | None = None
    exact: bool = False
    acorn: bool = False


# ── Filter expression leaf nodes ──────────────────────────────────────────────

@dataclass(frozen=True)
class CompareExpr:
    """field op literal  — covers =, !=, >, >=, <, <="""
    field: str
    op: str   # one of: "=", "!=", ">", ">=", "<", "<="
    value: str | int | float


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
    values: tuple[str | int | float, ...]


@dataclass(frozen=True)
class NotInExpr:
    """field NOT IN (v1, v2, ...)"""
    field: str
    values: tuple[str | int | float, ...]


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


@dataclass(frozen=True)
class InsertBulkStmt:
    collection: str
    values_list: tuple[dict[str, Any], ...]  # each dict must contain "text"
    model: str | None                         # dense model; None → use config default
    hybrid: bool = False
    sparse_model: str | None = None


@dataclass(frozen=True)
class CreateCollectionStmt:
    collection: str
    hybrid: bool = False    # if True, create with dense + sparse named vectors
    model: str | None = None  # dense model; None → use config default


@dataclass(frozen=True)
class CreateIndexStmt:
    collection: str
    field_name: str
    schema: str


@dataclass(frozen=True)
class DropCollectionStmt:
    collection: str


@dataclass(frozen=True)
class ShowCollectionsStmt:
    pass


@dataclass(frozen=True)
class SearchStmt:
    collection: str
    query_text: str
    limit: int
    model: str | None               # dense model; None → use config default
    hybrid: bool = False            # if True, use prefetch+RRF hybrid search
    sparse_only: bool = False       # if True, query only the sparse vector (no dense)
    sparse_model: str | None = None # sparse model for hybrid/sparse-only; None → SparseEmbedder.DEFAULT_MODEL
    query_filter: FilterExpr | None = None  # optional WHERE clause; default keeps existing tests valid
    rerank: bool = False                    # if True, apply cross-encoder reranking post-Qdrant
    rerank_model: str | None = None         # cross-encoder model; None → CrossEncoderEmbedder.DEFAULT_MODEL
    with_clause: SearchWith | None = None


@dataclass(frozen=True)
class RecommendStmt:
    collection: str
    positive_ids: tuple[str | int, ...]
    negative_ids: tuple[str | int, ...] = ()
    limit: int = 10
    strategy: str | None = None
    query_filter: FilterExpr | None = None
    offset: int = 0
    score_threshold: float | None = None
    with_clause: SearchWith | None = None
    lookup_from: tuple[str, str | None] | None = None
    using: str | None = None


@dataclass(frozen=True)
class DeleteStmt:
    collection: str
    point_id: str | int | None = None
    query_filter: FilterExpr | None = None


# Union type for all top-level statement nodes
ASTNode = (
    InsertStmt
    | InsertBulkStmt
    | CreateCollectionStmt
    | CreateIndexStmt
    | DropCollectionStmt
    | ShowCollectionsStmt
    | SearchStmt
    | RecommendStmt
    | DeleteStmt
)
