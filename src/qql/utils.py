from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

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
    Range,
    RecommendStrategy,
)

from .ast_nodes import (
    ASTNode,
    AndExpr,
    BetweenExpr,
    CompareExpr,
    FilterExpr,
    InExpr,
    InsertBulkStmt,
    InsertStmt,
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
    RecommendStmt,
    SearchStmt,
    SearchWith,
)
from .exceptions import QQLRuntimeError, QQLSyntaxError
from .lexer import TokenKind

_HYBRID_FUSION_VALUES = {"rrf", "dbsf"}


@dataclass(frozen=True)
class BatchGroup:
    kind: str
    collection: str | None
    statements: list[ASTNode]


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


def render_parameterized_query(template: str, params: dict[str, Any]) -> str:
    rendered = []
    in_string = False
    quote_char = ""
    i = 0
    while i < len(template):
        ch = template[i]
        if in_string:
            rendered.append(ch)
            if ch == "\\" and i + 1 < len(template):
                rendered.append(template[i + 1])
                i += 2
                continue
            if ch == quote_char:
                in_string = False
                quote_char = ""
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            rendered.append(ch)
            i += 1
            continue

        if ch == ":":
            name_start = i + 1
            name_end = name_start
            while name_end < len(template) and (
                template[name_end].isalnum() or template[name_end] == "_"
            ):
                name_end += 1
            name = template[name_start:name_end]
            if name in params:
                rendered.append(_qql_literal(params[name]))
                i = name_end
                continue

        rendered.append(ch)
        i += 1

    return "".join(rendered)


def _qql_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return _qql_string_literal(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_qql_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_qql_string_literal(str(key))}: {_qql_literal(item)}"
            for key, item in value.items()
        )
        return "{" + items + "}"
    return str(value)


def _qql_string_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f"'{escaped}'"


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


def _append_batch_group(
    groups: list[BatchGroup],
    kind: str | None,
    collection: str | None,
    statements: list[ASTNode],
) -> None:
    if statements:
        groups.append(BatchGroup(kind or "other", collection, statements))


def _compatible_insert_batch(current_group: list[ASTNode], stmt: InsertStmt) -> bool:
    if not current_group:
        return False
    prev_stmt = current_group[0]
    if not isinstance(prev_stmt, InsertStmt):
        return False
    return (
        stmt.model == prev_stmt.model
        and stmt.hybrid == prev_stmt.hybrid
        and stmt.sparse_model == prev_stmt.sparse_model
        and stmt.dense_vector == prev_stmt.dense_vector
        and stmt.sparse_vector == prev_stmt.sparse_vector
    )


def group_batch_statements(statements: tuple[ASTNode, ...]) -> list[BatchGroup]:
    groups: list[BatchGroup] = []
    current_type: str | None = None
    current_collection: str | None = None
    current_group: list[ASTNode] = []

    for stmt in statements:
        if isinstance(stmt, SearchStmt) and stmt.group_by is not None:
            _append_batch_group(groups, current_type, current_collection, current_group)
            groups.append(BatchGroup("other", stmt.collection, [stmt]))
            current_type = None
            current_collection = None
            current_group = []
            continue

        if isinstance(stmt, (SearchStmt, RecommendStmt)):
            coll = stmt.collection
            if current_type == "query" and current_collection == coll:
                current_group.append(stmt)
                continue
            _append_batch_group(groups, current_type, current_collection, current_group)
            current_type = "query"
            current_collection = coll
            current_group = [stmt]
            continue

        if isinstance(stmt, InsertStmt):
            coll = stmt.collection
            if (
                current_type == "insert"
                and current_collection == coll
                and _compatible_insert_batch(current_group, stmt)
            ):
                current_group.append(stmt)
                continue
            _append_batch_group(groups, current_type, current_collection, current_group)
            current_type = "insert"
            current_collection = coll
            current_group = [stmt]
            continue

        _append_batch_group(groups, current_type, current_collection, current_group)
        groups.append(BatchGroup("other", None, [stmt]))
        current_type = None
        current_collection = None
        current_group = []

    _append_batch_group(groups, current_type, current_collection, current_group)
    return groups


def build_bulk_insert_from_group(
    collection: str,
    statements: list[ASTNode],
) -> InsertBulkStmt:
    first = statements[0]
    if not isinstance(first, InsertStmt):
        raise QQLRuntimeError("Batch insert group must contain INSERT statements")
    insert_statements = [stmt for stmt in statements if isinstance(stmt, InsertStmt)]
    return InsertBulkStmt(
        collection=collection,
        values_list=tuple(stmt.values for stmt in insert_statements),
        model=first.model,
        hybrid=first.hybrid,
        sparse_model=first.sparse_model,
        dense_vector=first.dense_vector,
        sparse_vector=first.sparse_vector,
    )


def inserted_point_results(
    result: Any,
    statements: list[ASTNode],
    result_type: type,
) -> list[Any]:
    if not result.success:
        return [result_type(success=False, message=result.message) for _ in statements]

    inserted_ids = result.data.get("ids", []) if isinstance(result.data, dict) else []
    is_hybrid = "hybrid" in result.message
    label = "hybrid, batched" if is_hybrid else "batched"
    rows = []
    for idx, stmt in enumerate(statements):
        if not isinstance(stmt, InsertStmt):
            continue
        point_id = inserted_ids[idx] if idx < len(inserted_ids) else "unknown"
        rows.append(
            result_type(
                success=True,
                message=f"Inserted 1 point [{point_id}] ({label})",
                data={"id": point_id, "collection": stmt.collection},
            )
        )
    return rows


def build_qdrant_filter(expr: FilterExpr) -> Any:
    if isinstance(expr, AndExpr):
        return Filter(must=[build_qdrant_filter(op) for op in expr.operands])
    if isinstance(expr, OrExpr):
        return Filter(should=[build_qdrant_filter(op) for op in expr.operands])
    if isinstance(expr, NotExpr):
        return Filter(must_not=[build_qdrant_filter(expr.operand)])
    if isinstance(expr, CompareExpr):
        if expr.value is None:
            null_condition = IsNullCondition(is_null=PayloadField(key=expr.field))
            if expr.op == "=":
                return null_condition
            if expr.op == "!=":
                return Filter(must_not=[null_condition])
            raise QQLRuntimeError(
                f"Cannot use operator '{expr.op}' with null for field '{expr.field}'"
            )
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
        non_nulls = [value for value in expr.values if value is not None]
        null_condition = IsNullCondition(is_null=PayloadField(key=expr.field))
        if len(non_nulls) == len(expr.values):
            return FieldCondition(key=expr.field, match=MatchAny(any=non_nulls))
        if not non_nulls:
            return null_condition
        return Filter(
            should=[
                null_condition,
                FieldCondition(key=expr.field, match=MatchAny(any=non_nulls)),
            ]
        )
    if isinstance(expr, NotInExpr):
        non_nulls = [value for value in expr.values if value is not None]
        null_condition = IsNullCondition(is_null=PayloadField(key=expr.field))
        if len(non_nulls) != len(expr.values):
            must_not = [null_condition]
            if non_nulls:
                must_not.append(
                    FieldCondition(key=expr.field, match=MatchAny(any=non_nulls))
                )
            return Filter(must_not=must_not)
        return FieldCondition(
            key=expr.field,
            match=MatchExcept(**{"except": non_nulls}),
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
        hnsw_ef=override.hnsw_ef if override.hnsw_ef is not None else base.hnsw_ef,
        exact=override.exact if override.exact is not None else base.exact,
        acorn=override.acorn if override.acorn is not None else base.acorn,
        indexed_only=(
            override.indexed_only
            if override.indexed_only is not None
            else base.indexed_only
        ),
        quantization=(
            override.quantization
            if override.quantization is not None
            else base.quantization
        ),
        mmr_diversity=(
            override.mmr_diversity
            if override.mmr_diversity is not None
            else base.mmr_diversity
        ),
        mmr_candidates=(
            override.mmr_candidates
            if override.mmr_candidates is not None
            else base.mmr_candidates
        ),
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
