from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    AcornSearchParams,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HasIdCondition,
    IsEmptyCondition,
    IsNullCondition,
    LookupLocation,
    MatchAny,
    MatchExcept,
    MatchPhrase,
    MatchText,
    MatchTextAny,
    MatchValue,
    Modifier,
    PayloadField,
    PointStruct,
    Prefetch,
    Range,
    RecommendInput,
    RecommendQuery,
    RecommendStrategy,
    SearchParams,
    SparseVector,
    SparseVectorParams,
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
    ShowCollectionsStmt,
)
from .config import QQLConfig
from .embedder import CrossEncoderEmbedder, Embedder, SparseEmbedder

_RERANK_FETCH_MULTIPLIER = 4
_COLLECTION_VISIBILITY_TIMEOUT_SECONDS = 5.0
_COLLECTION_VISIBILITY_POLL_SECONDS = 0.05
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
        if isinstance(node, InsertBulkStmt):
            return self._execute_insert_bulk(node)
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
        if isinstance(node, RecommendStmt):
            return self._execute_recommend(node)
        if isinstance(node, DeleteStmt):
            return self._execute_delete(node)
        raise QQLRuntimeError(f"Unknown AST node type: {type(node)}")

    # ── Statement executors ───────────────────────────────────────────────

    def _execute_insert(self, node: InsertStmt) -> ExecutionResult:
        if "text" not in node.values:
            raise QQLRuntimeError("INSERT requires a 'text' field in VALUES")

        # Auto-detect hybrid when the user omitted USING HYBRID but the
        # collection already exists as a hybrid (named-vector) collection.
        use_hybrid = node.hybrid or self._collection_is_hybrid(node.collection)

        # ── Hybrid INSERT: dense + sparse vectors ──────────────────────────
        if use_hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_embedder = Embedder(dense_model)
            sparse_embedder = SparseEmbedder(sparse_model_name)

            dense_vector = dense_embedder.embed(node.values["text"])
            sparse_obj = sparse_embedder.embed(node.values["text"])
            sparse_vector = SparseVector(
                indices=sparse_obj["indices"],
                values=sparse_obj["values"],
            )

            # Auto-create hybrid collection if it doesn't exist yet
            if not self._client.collection_exists(node.collection):
                self._create_collection_and_wait(
                    collection_name=node.collection,
                    vectors_config={
                        "dense": VectorParams(
                            size=len(dense_vector), distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(modifier=Modifier.IDF)
                    },
                )

            point_id, payload = self._extract_point_id_and_payload(node.values)
            try:
                self._client.upsert(
                    collection_name=node.collection,
                    wait=True,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector={"dense": dense_vector, "sparse": sparse_vector},
                            payload=payload,
                        )
                    ],
                )
            except UnexpectedResponse as e:
                raise QQLRuntimeError(f"Qdrant error during INSERT: {e}") from e

            return ExecutionResult(
                success=True,
                message=f"Inserted 1 point [{point_id}] (hybrid)",
                data={"id": point_id, "collection": node.collection},
            )

        # ── Standard dense-only INSERT ─────────────────────────────────────
        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.values["text"])

        self._ensure_collection(node.collection, len(vector))

        point_id, payload = self._extract_point_id_and_payload(node.values)

        try:
            self._client.upsert(
                collection_name=node.collection,
                wait=True,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during INSERT: {e}") from e

        return ExecutionResult(
            success=True,
            message=f"Inserted 1 point [{point_id}]",
            data={"id": point_id, "collection": node.collection},
        )

    def _execute_insert_bulk(self, node: InsertBulkStmt) -> ExecutionResult:
        if not node.values_list:
            raise QQLRuntimeError("INSERT BULK VALUES list is empty")
        for i, vals in enumerate(node.values_list):
            if "text" not in vals:
                raise QQLRuntimeError(
                    f"INSERT BULK: item at index {i} is missing required 'text' field"
                )

        # Auto-detect hybrid when the user omitted USING HYBRID but the
        # collection already exists as a hybrid (named-vector) collection.
        use_hybrid = node.hybrid or self._collection_is_hybrid(node.collection)

        # ── Hybrid bulk INSERT: dense + sparse vectors ─────────────────────
        if use_hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_embedder = Embedder(dense_model)
            sparse_embedder = SparseEmbedder(sparse_model_name)

            points: list[PointStruct] = []
            for vals in node.values_list:
                point_id, payload = self._extract_point_id_and_payload(vals)
                dense_vector = dense_embedder.embed(vals["text"])
                sparse_obj = sparse_embedder.embed(vals["text"])
                sparse_vector = SparseVector(
                    indices=sparse_obj["indices"], values=sparse_obj["values"]
                )
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={"dense": dense_vector, "sparse": sparse_vector},
                        payload=payload,
                    )
                )

            if not self._client.collection_exists(node.collection):
                first_dense = dense_embedder.embed(node.values_list[0]["text"])
                self._create_collection_and_wait(
                    collection_name=node.collection,
                    vectors_config={
                        "dense": VectorParams(size=len(first_dense), distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(modifier=Modifier.IDF)
                    },
                )

            try:
                self._client.upsert(
                    collection_name=node.collection,
                    wait=True,
                    points=points,
                )
            except UnexpectedResponse as e:
                raise QQLRuntimeError(f"Qdrant error during INSERT BULK: {e}") from e

            return ExecutionResult(
                success=True,
                message=f"Inserted {len(points)} points (hybrid)",
            )

        # ── Standard dense-only bulk INSERT ───────────────────────────────
        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)

        points = []
        for vals in node.values_list:
            vector = embedder.embed(vals["text"])
            point_id, payload = self._extract_point_id_and_payload(vals)
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

        self._ensure_collection(node.collection, len(points[0].vector))

        try:
            self._client.upsert(
                collection_name=node.collection,
                wait=True,
                points=points,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during INSERT BULK: {e}") from e

        return ExecutionResult(
            success=True,
            message=f"Inserted {len(points)} points",
        )

    def _execute_create(self, node: CreateCollectionStmt) -> ExecutionResult:
        if self._client.collection_exists(node.collection):
            return ExecutionResult(
                success=True,
                message=f"Collection '{node.collection}' already exists",
            )

        dense_model_name = node.model or self._config.default_model

        # ── Hybrid collection: named dense + sparse vectors ────────────────
        if node.hybrid:
            embedder = Embedder(dense_model_name)
            dims = embedder.dimensions
            self._create_collection_and_wait(
                collection_name=node.collection,
                vectors_config={
                    "dense": VectorParams(size=dims, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF)
                },
            )
            return ExecutionResult(
                success=True,
                message=(
                    f"Collection '{node.collection}' created "
                    f"(hybrid: {dims}-dim dense + BM25 sparse, cosine distance)"
                ),
            )

        # ── Standard dense-only collection ─────────────────────────────────
        embedder = Embedder(dense_model_name)
        dims = embedder.dimensions
        self._create_collection_and_wait(
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

        # Build WHERE filter (shared by both hybrid and dense-only paths)
        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        search_params = self._build_search_params(node.with_clause)

        # When reranking is requested, fetch more candidates so the reranker has
        # enough material to reorder; only `node.limit` results are returned.
        fetch_limit = node.limit * _RERANK_FETCH_MULTIPLIER if node.rerank else node.limit

        # ── Hybrid SEARCH: prefetch dense+sparse, fuse with RRF ───────────
        if node.hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_embedder = Embedder(dense_model)
            sparse_embedder = SparseEmbedder(sparse_model_name)

            dense_vector = dense_embedder.embed(node.query_text)
            sparse_obj = sparse_embedder.query_embed(node.query_text)
            sparse_vector = SparseVector(
                indices=sparse_obj["indices"],
                values=sparse_obj["values"],
            )

            try:
                response = self._client.query_points(
                    collection_name=node.collection,
                    prefetch=[
                        Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=node.limit * 4,
                            params=search_params,
                        ),
                        Prefetch(
                            query=sparse_vector,
                            using="sparse",
                            limit=node.limit * 4,
                            params=search_params,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=fetch_limit,
                    query_filter=qdrant_filter,
                )
            except UnexpectedResponse as e:
                raise QQLRuntimeError(f"Qdrant error during SEARCH: {e}") from e

            results = [
                {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
                for h in response.points
            ]

            if node.rerank:
                results = self._apply_reranking(node.query_text, results, node.limit, node.rerank_model)
                return ExecutionResult(
                    success=True,
                    message=f"Found {len(results)} result(s) (hybrid, reranked)",
                    data=results,
                )

            return ExecutionResult(
                success=True,
                message=f"Found {len(results)} result(s) (hybrid)",
                data=results,
            )

        # ── Sparse-only SEARCH: query the "sparse" named vector directly ─────
        if node.sparse_only:
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            sparse_embedder = SparseEmbedder(sparse_model_name)
            sparse_obj = sparse_embedder.query_embed(node.query_text)
            sparse_vector = SparseVector(
                indices=sparse_obj["indices"],
                values=sparse_obj["values"],
            )

            try:
                response = self._client.query_points(
                    collection_name=node.collection,
                    query=sparse_vector,
                    using="sparse",
                    limit=fetch_limit,
                    query_filter=qdrant_filter,
                )
            except UnexpectedResponse as e:
                raise QQLRuntimeError(f"Qdrant error during SEARCH: {e}") from e

            results = [
                {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
                for h in response.points
            ]

            if node.rerank:
                results = self._apply_reranking(node.query_text, results, node.limit, node.rerank_model)
                return ExecutionResult(
                    success=True,
                    message=f"Found {len(results)} result(s) (sparse, reranked)",
                    data=results,
                )

            return ExecutionResult(
                success=True,
                message=f"Found {len(results)} result(s) (sparse)",
                data=results,
            )

        # ── Standard dense-only SEARCH ─────────────────────────────────────
        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.query_text)

        try:
            query_using = self._get_dense_vector_name(node.collection)
            response = self._client.query_points(
                collection_name=node.collection,
                query=vector,
                using=query_using,
                limit=fetch_limit,
                query_filter=qdrant_filter,
                search_params=search_params,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during SEARCH: {e}") from e

        results = [
            {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
            for h in response.points
        ]

        if node.rerank:
            results = self._apply_reranking(node.query_text, results, node.limit, node.rerank_model)
            return ExecutionResult(
                success=True,
                message=f"Found {len(results)} result(s) (reranked)",
                data=results,
            )

        return ExecutionResult(
            success=True,
            message=f"Found {len(results)} result(s)",
            data=results,
        )

    def _execute_recommend(self, node: RecommendStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )
        qdrant_filter = self._exclude_ids_from_filter(
            qdrant_filter,
            [*node.positive_ids, *node.negative_ids],
        )

        recommend_input = RecommendInput(
            positive=list(node.positive_ids),
            negative=list(node.negative_ids) or None,
            strategy=self._parse_recommend_strategy(node.strategy),
        )

        search_params = self._build_search_params(node.with_clause)

        lookup_from: LookupLocation | None = None
        if node.lookup_from is not None:
            lookup_from = LookupLocation(
                collection=node.lookup_from[0],
                vector=node.lookup_from[1],
            )

        try:
            response = self._client.query_points(
                collection_name=node.collection,
                query=RecommendQuery(recommend=recommend_input),
                limit=node.limit,
                offset=node.offset or None,
                query_filter=qdrant_filter,
                search_params=search_params,
                score_threshold=node.score_threshold,
                using=node.using,
                lookup_from=lookup_from,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during RECOMMEND: {e}") from e

        results = [
            {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
            for h in response.points
        ]

        return ExecutionResult(
            success=True,
            message=f"Found {len(results)} recommendation(s)",
            data=results,
        )

    def _build_search_params(self, with_clause: SearchWith | None) -> SearchParams | None:
        if with_clause is None:
            return None
        return SearchParams(
            hnsw_ef=with_clause.hnsw_ef,
            exact=with_clause.exact,
            acorn=AcornSearchParams(enable=True) if with_clause.acorn else None,
        )

    def _parse_recommend_strategy(
        self, strategy: str | None
    ) -> RecommendStrategy | None:
        if strategy is None:
            return None
        try:
            return RecommendStrategy(strategy)
        except ValueError as e:
            raise QQLRuntimeError(
                "Unknown recommend strategy "
                f"'{strategy}'. Expected one of: average_vector, best_score, sum_scores"
            ) from e

    def _exclude_ids_from_filter(
        self,
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

    def _extract_point_id_and_payload(
        self, values: dict[str, Any]
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

    def _get_dense_vector_name(self, collection_name: str) -> str | None:
        """Return the dense vector name for named-vector collections.

        Dense-only QQL searches should keep working against hybrid collections,
        which store vectors under the explicit ``dense`` name.
        """
        info = self._client.get_collection(collection_name)
        vectors = info.config.params.vectors  # type: ignore[union-attr]
        if isinstance(vectors, dict):
            return "dense"
        return None

    def _apply_reranking(
        self,
        query: str,
        results: list[dict],
        limit: int,
        rerank_model: str | None,
    ) -> list[dict]:
        """Re-score candidates with a cross-encoder and return top-``limit`` results."""
        model_name = rerank_model or CrossEncoderEmbedder.DEFAULT_MODEL
        reranker = CrossEncoderEmbedder(model_name)
        texts = [r["payload"].get("text", "") for r in results]
        scores = reranker.rerank(query, texts)
        for r, s in zip(results, scores):
            r["score"] = round(float(s), 4)
        return sorted(results, key=lambda r: r["score"], reverse=True)[:limit]

    def _execute_delete(self, node: DeleteStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=node.collection,
                wait=True,
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

    def _collection_is_hybrid(self, name: str) -> bool:
        """Return True if *name* exists and uses named vectors (hybrid collection)."""
        if not self._client.collection_exists(name):
            return False
        info = self._client.get_collection(name)
        vectors = info.config.params.vectors  # type: ignore[union-attr]
        return isinstance(vectors, dict)

    def _ensure_collection(self, name: str, vector_size: int) -> None:
        """Create the collection if it doesn't exist. Raises on dimension mismatch.

        For named-vector (hybrid) collections the validation is skipped — those
        collections are managed directly by the hybrid insert/create paths.
        """
        if self._client.collection_exists(name):
            info = self._client.get_collection(name)
            vectors = info.config.params.vectors  # type: ignore[union-attr]
            if isinstance(vectors, dict):
                # Named-vector (hybrid) collection — skip validation here;
                # the hybrid insert path manages its own collection creation.
                pass
            else:
                # Unnamed single-vector collection: validate dimensions
                if vectors.size != vector_size:
                    raise QQLRuntimeError(
                        f"Vector dimension mismatch: collection '{name}' expects "
                        f"{vectors.size} dims, but model produces {vector_size} dims. "
                        f"Specify a compatible model with USING MODEL '<model>'."
                    )
        else:
            self._create_collection_and_wait(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def _create_collection_and_wait(self, **kwargs: Any) -> None:
        collection_name = kwargs["collection_name"]
        self._client.create_collection(**kwargs)

        deadline = time.monotonic() + _COLLECTION_VISIBILITY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._client.collection_exists(collection_name):
                return
            time.sleep(_COLLECTION_VISIBILITY_POLL_SECONDS)

        raise QQLRuntimeError(
            f"Collection '{collection_name}' was created but did not become visible in time"
        )
