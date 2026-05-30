from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from grpc import RpcError, StatusCode
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    HasIdCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchExcept,
    MatchPhrase,
    MatchText,
    MatchTextAny,
    MatchValue,
    Mmr,
    NearestQuery,
    PayloadField,
    Prefetch,
    Range,
    RecommendStrategy,
)

from .ast_nodes import (
    AndExpr,
    BetweenExpr,
    CompareExpr,
    FilterExpr,
    InExpr,
    IsEmptyExpr,
    IsNotEmptyExpr,
    IsNotNullExpr,
    IsNullExpr,
    MatchAnyExpr,
    MatchPhraseExpr,
    MatchTextExpr,
    NotExpr,
    NotInExpr,
    OrExpr,
    SearchStmt,
    SearchWith,
)
from .exceptions import QQLRuntimeError, QQLSyntaxError
from .lexer import TokenKind

_HYBRID_FUSION_VALUES = {"rrf", "dbsf"}


def is_grpc_not_found_error(error: BaseException) -> bool:
    return isinstance(error, RpcError) and error.code() == StatusCode.NOT_FOUND


@dataclass(frozen=True)
class SearchUsingOptions:
    model: str | None = None
    hybrid: bool = False
    fusion: str | None = None
    sparse_only: bool = False
    sparse_model: str | None = None
    dense_vector: str | None = None
    sparse_vector: str | None = None


@dataclass(frozen=True)
class SearchGroupByOptions:
    group_by: str | None = None
    group_size: int = 3


def collection_topology_kwargs(vectors: Any, sparse_vectors: Any) -> dict[str, Any]:
    if isinstance(vectors, dict):
        dense_names = tuple(vectors.keys())
        dense_sizes = tuple(
            (k, v.size)
            for k, v in vectors.items()
            if getattr(v, "size", None) is not None
        )
        has_unnamed_dense = False
        is_named_dense = True
    elif vectors is None:
        dense_names = ()
        dense_sizes = ()
        has_unnamed_dense = False
        is_named_dense = False
    else:
        dense_names = ()
        unnamed_size = getattr(vectors, "size", None)
        dense_sizes = (("", unnamed_size),) if unnamed_size is not None else ()
        has_unnamed_dense = True
        is_named_dense = False

    sparse_names = tuple(sparse_vectors.keys()) if isinstance(sparse_vectors, dict) else ()
    return {
        "exists": True,
        "is_named_dense": is_named_dense,
        "has_unnamed_dense": has_unnamed_dense,
        "dense_names": dense_names,
        "sparse_names": sparse_names,
        "dense_sizes": dense_sizes,
    }


def build_qdrant_filter(expr: FilterExpr) -> Any:
    if isinstance(expr, AndExpr):
        return Filter(must=[build_qdrant_filter(op) for op in expr.operands])
    if isinstance(expr, OrExpr):
        return Filter(should=[build_qdrant_filter(op) for op in expr.operands])
    if isinstance(expr, NotExpr):
        return Filter(must_not=[build_qdrant_filter(expr.operand)])
    if isinstance(expr, CompareExpr):
        if expr.op == "=":
            return FieldCondition(key=expr.field, match=MatchValue(value=expr.value))
        if expr.op == "!=":
            return Filter(
                must_not=[
                    FieldCondition(key=expr.field, match=MatchValue(value=expr.value))
                ]
            )
        range_key = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[expr.op]
        return FieldCondition(key=expr.field, range=Range(**{range_key: expr.value}))
    if isinstance(expr, BetweenExpr):
        return FieldCondition(key=expr.field, range=Range(gte=expr.low, lte=expr.high))
    if isinstance(expr, InExpr):
        return FieldCondition(key=expr.field, match=MatchAny(any=list(expr.values)))
    if isinstance(expr, NotInExpr):
        return FieldCondition(
            key=expr.field,
            match=MatchExcept(**{"except": list(expr.values)}),
        )
    if isinstance(expr, IsNullExpr):
        return IsNullCondition(is_null=PayloadField(key=expr.field))
    if isinstance(expr, IsNotNullExpr):
        return Filter(must_not=[IsNullCondition(is_null=PayloadField(key=expr.field))])
    if isinstance(expr, IsEmptyExpr):
        return IsEmptyCondition(is_empty=PayloadField(key=expr.field))
    if isinstance(expr, IsNotEmptyExpr):
        return Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key=expr.field))])
    if isinstance(expr, MatchTextExpr):
        return FieldCondition(key=expr.field, match=MatchText(text=expr.text))
    if isinstance(expr, MatchAnyExpr):
        return FieldCondition(key=expr.field, match=MatchTextAny(text_any=expr.text))
    if isinstance(expr, MatchPhraseExpr):
        return FieldCondition(key=expr.field, match=MatchPhrase(phrase=expr.text))
    raise QQLRuntimeError(f"Unknown filter expression type: {type(expr)}")


def wrap_as_filter(qdrant_expr: Any) -> Filter:
    if isinstance(qdrant_expr, Filter):
        return qdrant_expr
    return Filter(must=[qdrant_expr])


def resolve_hybrid_fusion(fusion: str | None) -> Fusion:
    if fusion is None or fusion == "rrf":
        return Fusion.RRF
    if fusion == "dbsf":
        return Fusion.DBSF
    raise QQLRuntimeError(
        f"Unsupported hybrid fusion '{fusion}'; expected 'rrf' or 'dbsf'"
    )


def has_mmr(with_clause: SearchWith | None) -> bool:
    return with_clause is not None and (
        with_clause.mmr_diversity is not None or with_clause.mmr_candidates is not None
    )


def validate_search_mmr_usage(node: SearchStmt) -> None:
    if not has_mmr(node.with_clause):
        return
    if node.sparse_only:
        raise QQLRuntimeError("MMR is not supported with USING SPARSE yet")


def build_dense_query(
    vector: list[float],
    with_clause: SearchWith | None,
) -> list[float] | NearestQuery:
    if not has_mmr(with_clause):
        return vector
    return NearestQuery(
        nearest=vector,
        mmr=Mmr(
            diversity=with_clause.mmr_diversity,
            candidates_limit=with_clause.mmr_candidates,
        ),
    )


def build_hybrid_prefetches(
    topology: Any,
    node: SearchStmt,
    dense_vector: list[float],
    sparse_vector: Any,
    search_params: Any,
    prefetch_multiplier: int,
) -> list[Prefetch]:
    prefetch_limit = node.limit * prefetch_multiplier
    return [
        Prefetch(
            query=build_dense_query(dense_vector, node.with_clause),
            using=topology.dense_using(node.dense_vector),
            limit=prefetch_limit,
            params=search_params,
        ),
        Prefetch(
            query=sparse_vector,
            using=topology.sparse_using(node.sparse_vector),
            limit=prefetch_limit,
            params=search_params,
        ),
    ]


def parse_recommend_strategy(strategy: str | None) -> RecommendStrategy | None:
    if strategy is None:
        return None
    try:
        return RecommendStrategy(strategy)
    except ValueError as e:
        raise QQLRuntimeError(
            "Unknown recommend strategy "
            f"'{strategy}'. Expected one of: average_vector, best_score, sum_scores"
        ) from e


def exclude_ids_from_filter(
    query_filter: Filter | None,
    point_ids: list[str | int],
) -> Filter | None:
    if not point_ids:
        return query_filter

    exclude_condition = HasIdCondition(has_id=point_ids)
    if query_filter is None:
        return Filter(must_not=[exclude_condition])

    return Filter(
        must=list(query_filter.must or []),
        should=list(query_filter.should or []),
        must_not=[*(query_filter.must_not or []), exclude_condition],
        min_should=query_filter.min_should,
    )


def extract_point_id_and_payload(
    values: dict[str, Any],
) -> tuple[str | int, dict[str, Any]]:
    payload = dict(values)
    if "id" not in payload:
        return str(uuid.uuid4()), payload

    point_id = payload.pop("id")
    if isinstance(point_id, bool):
        raise QQLRuntimeError(
            "INSERT id must be an unsigned integer or UUID string when provided"
        )
    if isinstance(point_id, int):
        if point_id < 0:
            raise QQLRuntimeError(
                "INSERT id must be an unsigned integer or UUID string when provided"
            )
        return point_id, payload
    if isinstance(point_id, str):
        try:
            uuid.UUID(point_id)
        except ValueError as e:
            raise QQLRuntimeError(
                "INSERT id must be an unsigned integer or UUID string when provided"
            ) from e
        return point_id, payload
    raise QQLRuntimeError(
        "INSERT id must be an unsigned integer or UUID string when provided"
    )


def build_dense_point_vector(
    topology: Any,
    vector: list[float],
    explicit_vector: str | None,
    default_dense_vector_name: str,
) -> list[float] | dict[str, list[float]]:
    if not topology.exists:
        return {explicit_vector or default_dense_vector_name: vector}
    vector_name = topology.dense_payload_name(explicit_vector)
    if vector_name is None:
        return vector
    return {vector_name: vector}


def merge_search_with(base: SearchWith | None, override: SearchWith) -> SearchWith:
    if base is None:
        return override
    return SearchWith(
        hnsw_ef=override.hnsw_ef or base.hnsw_ef,
        exact=override.exact or base.exact,
        acorn=override.acorn or base.acorn,
        indexed_only=override.indexed_only or base.indexed_only,
        quantization=override.quantization or base.quantization,
        mmr_diversity=(
            override.mmr_diversity
            if override.mmr_diversity is not None
            else base.mmr_diversity
        ),
        mmr_candidates=override.mmr_candidates or base.mmr_candidates,
    )


def parse_search_lookup(parser: Any) -> tuple[str, str | None] | None:
    if parser._peek().kind != TokenKind.LOOKUP:
        return None
    parser._advance()
    parser._expect(TokenKind.FROM)
    lookup_collection = parser._parse_identifier()
    lookup_vector: str | None = None
    if parser._peek().kind == TokenKind.VECTOR:
        parser._advance()
        lookup_vector = parser._expect(TokenKind.STRING).value
    return lookup_collection, lookup_vector


def parse_search_using(parser: Any) -> SearchUsingOptions:
    if parser._peek().kind != TokenKind.USING:
        return SearchUsingOptions()

    parser._advance()
    if parser._peek().kind == TokenKind.HYBRID:
        return _parse_hybrid_using(parser)
    if parser._peek().kind == TokenKind.SPARSE:
        return _parse_sparse_using(parser)
    if parser._peek().kind == TokenKind.VECTOR:
        parser._advance()
        return SearchUsingOptions(dense_vector=parser._expect(TokenKind.STRING).value)

    parser._expect(TokenKind.MODEL)
    return SearchUsingOptions(model=parser._expect(TokenKind.STRING).value)


def _parse_hybrid_using(parser: Any) -> SearchUsingOptions:
    parser._advance()
    model: str | None = None
    fusion: str | None = None
    sparse_model: str | None = None
    dense_vector: str | None = None
    sparse_vector: str | None = None

    while parser._peek().kind in (TokenKind.FUSION, TokenKind.DENSE, TokenKind.SPARSE):
        sub = parser._advance()
        if sub.kind == TokenKind.FUSION:
            value_tok = parser._expect(TokenKind.STRING)
            fusion = value_tok.value.lower()
            if fusion not in _HYBRID_FUSION_VALUES:
                raise QQLSyntaxError(
                    f"Unsupported hybrid fusion '{value_tok.value}'; expected 'rrf' or 'dbsf'",
                    value_tok.pos,
                )
            continue
        if parser._peek().kind == TokenKind.MODEL:
            parser._advance()
            parsed_model = parser._expect(TokenKind.STRING).value
            if sub.kind == TokenKind.DENSE:
                model = parsed_model
            else:
                sparse_model = parsed_model
            continue
        if parser._peek().kind == TokenKind.VECTOR:
            parser._advance()
            name = parser._expect(TokenKind.STRING).value
            if sub.kind == TokenKind.DENSE:
                dense_vector = name
            else:
                sparse_vector = name
            continue
        raise QQLSyntaxError(
            "Expected MODEL or VECTOR after DENSE/SPARSE in USING HYBRID",
            parser._peek().pos,
        )

    return SearchUsingOptions(
        model=model,
        hybrid=True,
        fusion=fusion,
        sparse_model=sparse_model,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
    )


def _parse_sparse_using(parser: Any) -> SearchUsingOptions:
    parser._advance()
    sparse_model: str | None = None
    sparse_vector: str | None = None
    while parser._peek().kind in (TokenKind.MODEL, TokenKind.VECTOR):
        sub = parser._advance()
        if sub.kind == TokenKind.MODEL:
            sparse_model = parser._expect(TokenKind.STRING).value
        else:
            sparse_vector = parser._expect(TokenKind.STRING).value
    return SearchUsingOptions(
        sparse_only=True,
        sparse_model=sparse_model,
        sparse_vector=sparse_vector,
    )


def parse_search_with(parser: Any, with_clause: SearchWith | None) -> SearchWith | None:
    if parser._peek().kind == TokenKind.EXACT:
        parser._advance()
        with_clause = merge_search_with(with_clause, SearchWith(exact=True))

    if parser._peek().kind == TokenKind.WITH:
        parser._advance()
        with_clause = merge_search_with(with_clause, parser._parse_with_clause())

    return with_clause


def parse_search_group_by(
    parser: Any,
    offset: int,
    rerank: bool,
) -> SearchGroupByOptions:
    if parser._peek().kind != TokenKind.GROUP:
        return SearchGroupByOptions()

    if offset > 0:
        raise QQLSyntaxError("OFFSET cannot be used with GROUP BY", parser._peek().pos)
    parser._advance()
    parser._expect(TokenKind.BY)
    group_by = parser._parse_field_path()
    if rerank:
        raise QQLSyntaxError(
            "GROUP BY and RERANK cannot be combined in the same SEARCH statement",
            parser._peek().pos,
        )

    group_size = 3
    if parser._peek().kind == TokenKind.GROUP_SIZE:
        parser._advance()
        group_size_tok = parser._peek()
        group_size = int(parser._expect(TokenKind.INTEGER).value)
        if group_size <= 0:
            raise QQLSyntaxError(
                f"GROUP_SIZE must be a positive integer, got {group_size}",
                group_size_tok.pos,
            )
    return SearchGroupByOptions(group_by=group_by, group_size=group_size)
