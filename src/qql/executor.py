from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchExcept,
    MatchPhrase,
    MatchText,
    MatchTextAny,
    MatchValue,
    PayloadField,
    PointStruct,
    Range,
    VectorParams,
)

from .ast_nodes import (
    ASTNode,
    AndExpr,
    BetweenExpr,
    CompareExpr,
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    FilterExpr,
    InExpr,
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
    SearchStmt,
    ShowCollectionsStmt,
)
from .config import QQLConfig
from .embedder import Embedder
from .exceptions import QQLRuntimeError


@dataclass
class ExecutionResult:
    success: bool
    message: str
    data: Any = None


class Executor:
    def __init__(self, client: QdrantClient, config: QQLConfig) -> None:
        self._client = client
        self._config = config

    def execute(self, node: ASTNode) -> ExecutionResult:
        if isinstance(node, InsertStmt):
            return self._execute_insert(node)
        if isinstance(node, CreateCollectionStmt):
            return self._execute_create(node)
        if isinstance(node, DropCollectionStmt):
            return self._execute_drop(node)
        if isinstance(node, ShowCollectionsStmt):
            return self._execute_show(node)
        if isinstance(node, SearchStmt):
            return self._execute_search(node)
        if isinstance(node, DeleteStmt):
            return self._execute_delete(node)
        raise QQLRuntimeError(f"Unknown AST node type: {type(node)}")

    # ── Statement executors ───────────────────────────────────────────────

    def _execute_insert(self, node: InsertStmt) -> ExecutionResult:
        if "text" not in node.values:
            raise QQLRuntimeError("INSERT requires a 'text' field in VALUES")

        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.values["text"])

        self._ensure_collection(node.collection, len(vector))

        point_id = str(uuid.uuid4())
        payload = dict(node.values)

        try:
            self._client.upsert(
                collection_name=node.collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during INSERT: {e}") from e

        return ExecutionResult(
            success=True,
            message=f"Inserted 1 point [{point_id}]",
            data={"id": point_id, "collection": node.collection},
        )

    def _execute_create(self, node: CreateCollectionStmt) -> ExecutionResult:
        if self._client.collection_exists(node.collection):
            return ExecutionResult(
                success=True,
                message=f"Collection '{node.collection}' already exists",
            )
        embedder = Embedder(self._config.default_model)
        dims = embedder.dimensions
        self._client.create_collection(
            collection_name=node.collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' created ({dims}-dimensional vectors, cosine distance)",
        )

    def _execute_drop(self, node: DropCollectionStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        self._client.delete_collection(node.collection)
        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' dropped",
        )

    def _execute_show(self, node: ShowCollectionsStmt) -> ExecutionResult:
        response = self._client.get_collections()
        names = [c.name for c in response.collections]
        return ExecutionResult(
            success=True,
            message=f"{len(names)} collection(s) found",
            data=names,
        )

    def _execute_search(self, node: SearchStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.query_text)

        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        try:
            response = self._client.query_points(
                collection_name=node.collection,
                query=vector,
                limit=node.limit,
                query_filter=qdrant_filter,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during SEARCH: {e}") from e

        results = [
            {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
            for h in response.points
        ]
        return ExecutionResult(
            success=True,
            message=f"Found {len(results)} result(s)",
            data=results,
        )

    def _execute_delete(self, node: DeleteStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=node.collection,
                points_selector=PointIdsList(points=[node.point_id]),
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during DELETE: {e}") from e

        return ExecutionResult(
            success=True,
            message=f"Deleted point '{node.point_id}' from '{node.collection}'",
        )

    # ── Filter conversion ─────────────────────────────────────────────────

    def _build_qdrant_filter(self, expr: FilterExpr) -> Any:
        """Convert a FilterExpr AST node into a Qdrant model object.

        Returns one of: Filter, FieldCondition, IsNullCondition, IsEmptyCondition.
        Use _wrap_as_filter() to guarantee the top-level result is a Filter.
        """
        # ── Logical combinators ───────────────────────────────────────────
        if isinstance(expr, AndExpr):
            return Filter(must=[self._build_qdrant_filter(op) for op in expr.operands])

        if isinstance(expr, OrExpr):
            return Filter(should=[self._build_qdrant_filter(op) for op in expr.operands])

        if isinstance(expr, NotExpr):
            return Filter(must_not=[self._build_qdrant_filter(expr.operand)])

        # ── Comparison ────────────────────────────────────────────────────
        if isinstance(expr, CompareExpr):
            if expr.op == "=":
                return FieldCondition(
                    key=expr.field, match=MatchValue(value=expr.value)
                )
            if expr.op == "!=":
                return Filter(
                    must_not=[
                        FieldCondition(key=expr.field, match=MatchValue(value=expr.value))
                    ]
                )
            _range_key = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[expr.op]
            return FieldCondition(
                key=expr.field, range=Range(**{_range_key: expr.value})
            )

        # ── BETWEEN ───────────────────────────────────────────────────────
        if isinstance(expr, BetweenExpr):
            return FieldCondition(
                key=expr.field, range=Range(gte=expr.low, lte=expr.high)
            )

        # ── IN / NOT IN ───────────────────────────────────────────────────
        if isinstance(expr, InExpr):
            return FieldCondition(
                key=expr.field, match=MatchAny(any=list(expr.values))
            )

        if isinstance(expr, NotInExpr):
            return FieldCondition(
                key=expr.field,
                match=MatchExcept(**{"except": list(expr.values)}),
            )

        # ── IS NULL / IS NOT NULL ─────────────────────────────────────────
        if isinstance(expr, IsNullExpr):
            return IsNullCondition(is_null=PayloadField(key=expr.field))

        if isinstance(expr, IsNotNullExpr):
            return Filter(
                must_not=[IsNullCondition(is_null=PayloadField(key=expr.field))]
            )

        # ── IS EMPTY / IS NOT EMPTY ───────────────────────────────────────
        if isinstance(expr, IsEmptyExpr):
            return IsEmptyCondition(is_empty=PayloadField(key=expr.field))

        if isinstance(expr, IsNotEmptyExpr):
            return Filter(
                must_not=[IsEmptyCondition(is_empty=PayloadField(key=expr.field))]
            )

        # ── Full-text MATCH ───────────────────────────────────────────────
        if isinstance(expr, MatchTextExpr):
            return FieldCondition(key=expr.field, match=MatchText(text=expr.text))

        if isinstance(expr, MatchAnyExpr):
            return FieldCondition(
                key=expr.field, match=MatchTextAny(text_any=expr.text)
            )

        if isinstance(expr, MatchPhraseExpr):
            return FieldCondition(
                key=expr.field, match=MatchPhrase(phrase=expr.text)
            )

        raise QQLRuntimeError(f"Unknown filter expression type: {type(expr)}")

    def _wrap_as_filter(self, qdrant_expr: Any) -> Filter:
        """Ensure the top-level expression is a Filter (required by query_points)."""
        if isinstance(qdrant_expr, Filter):
            return qdrant_expr
        return Filter(must=[qdrant_expr])

    # ── Collection helpers ────────────────────────────────────────────────

    def _ensure_collection(self, name: str, vector_size: int) -> None:
        """Create the collection if it doesn't exist. Raises on dimension mismatch."""
        if self._client.collection_exists(name):
            info = self._client.get_collection(name)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != vector_size:
                raise QQLRuntimeError(
                    f"Vector dimension mismatch: collection '{name}' expects "
                    f"{existing_size} dims, but model produces {vector_size} dims. "
                    f"Specify a compatible model with USING MODEL '<model>'."
                )
        else:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
