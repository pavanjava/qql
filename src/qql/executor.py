from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    AcornSearchParams,
    BinaryQuantization,
    BinaryQuantizationConfig,
    CollectionParamsDiff,
    CompressionRatio,
    Distance,
    Disabled,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HasIdCondition,
    HnswConfigDiff,
    IsEmptyCondition,
    IsNullCondition,
    KeywordIndexParams,
    KeywordIndexType,
    Language,
    LookupLocation,
    MatchAny,
    MatchExcept,
    MatchPhrase,
    MatchText,
    MatchTextAny,
    MatchValue,
    Mmr,
    Modifier,
    NearestQuery,
    OptimizersConfigDiff,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    PointVectors,
    Prefetch,
    ProductQuantization,
    ProductQuantizationConfig,
    QuantizationSearchParams,
    Range,
    RecommendInput,
    RecommendQuery,
    RecommendStrategy,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    TurboQuantBitSize,
    TurboQuantization,
    TurboQuantQuantizationConfig,
    SearchParams,
    SparseVector,
    SparseVectorParams,
    StopwordsSet,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    UuidIndexParams,
    UuidIndexType,
    VectorParams,
    VectorParamsDiff,
    MaxOptimizationThreadsSetting,
)

from .ast_nodes import (
    ASTNode,
    AlterCollectionStmt,
    AndExpr,
    BetweenExpr,
    CollectionConfig,
    CompareExpr,
    CreateCollectionStmt,
    CreateIndexStmt,
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
    QuantizationUpdate,
    QuantizationConfig,
    QuantizationType,
    RecommendStmt,
    SelectStmt,
    ScrollStmt,
    SearchStmt,
    SearchWith,
    ShowCollectionStmt,
    ShowCollectionsStmt,
    UpdateVectorStmt,
    UpdatePayloadStmt,
)
from .config import QQLConfig
from .embedder import CrossEncoderEmbedder, Embedder, SparseEmbedder
from .exceptions import QQLRuntimeError

_RERANK_FETCH_MULTIPLIER = 4
_HYBRID_PREFETCH_MULTIPLIER = 4
_COLLECTION_VISIBILITY_TIMEOUT_SECONDS = 5.0
_COLLECTION_VISIBILITY_POLL_SECONDS = 0.05


@dataclass
class ExecutionResult:
    success: bool
    message: str
    data: Any = None


@dataclass(frozen=True)
class CollectionTopology:
    exists: bool
    is_named_dense: bool
    has_unnamed_dense: bool = False
    dense_names: tuple[str, ...] = ()
    sparse_names: tuple[str, ...] = ()
    # Sizes fetched once in _resolve_topology() so _ensure_collection() never
    # needs a second get_collection() call.
    dense_sizes: tuple[tuple[str, int], ...] = ()

    def dense_size_map(self) -> dict[str, int]:
        """Return {vector_name: size} for every dense vector whose size was fetched.

        Unnamed single-vector collections appear under the ``""`` key, matching
        ``dense_config_key()``.  Returns an empty dict when ``exists`` is False or
        sizes were not available (e.g. when a mock omits the size attribute).
        """
        return dict(self.dense_sizes)

    @property
    def has_dense(self) -> bool:
        return self.has_unnamed_dense or bool(self.dense_names)

    @property
    def has_sparse(self) -> bool:
        return bool(self.sparse_names)

    @property
    def is_hybrid(self) -> bool:
        return self.has_dense and self.has_sparse

    def dense_using(self, explicit: str | None = None) -> str | None:
        if explicit is not None:
            if self.exists and self.has_unnamed_dense:
                raise QQLRuntimeError(
                    "Collection uses an unnamed dense vector; omit USING VECTOR"
                )
            if self.exists and explicit not in self.dense_names:
                raise QQLRuntimeError(
                    f"Collection has no dense vector named '{explicit}'"
                )
            return explicit
        if self.has_unnamed_dense:
            return None
        if len(self.dense_names) == 1:
            return self.dense_names[0]
        if not self.dense_names:
            raise QQLRuntimeError("Collection has no dense vector")
        raise QQLRuntimeError(
            "Collection has multiple dense vectors; specify one with USING VECTOR '<name>'"
        )

    def dense_payload_name(self, explicit: str | None = None) -> str | None:
        return self.dense_using(explicit)

    def dense_config_key(self, explicit: str | None = None) -> str:
        name = self.dense_using(explicit)
        return "" if name is None else name

    def sparse_using(self, explicit: str | None = None) -> str:
        if explicit is not None:
            if self.exists and explicit not in self.sparse_names:
                raise QQLRuntimeError(
                    f"Collection has no sparse vector named '{explicit}'"
                )
            return explicit
        if len(self.sparse_names) == 1:
            return self.sparse_names[0]
        if not self.sparse_names:
            raise QQLRuntimeError("Collection has no sparse vector")
        raise QQLRuntimeError(
            "Collection has multiple sparse vectors; specify one with USING SPARSE VECTOR '<name>'"
        )


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
        if isinstance(node, AlterCollectionStmt):
            return self._execute_alter_collection(node)
        if isinstance(node, CreateIndexStmt):
            return self._execute_create_index(node)
        if isinstance(node, DropCollectionStmt):
            return self._execute_drop(node)
        if isinstance(node, ShowCollectionsStmt):
            return self._execute_show(node)
        if isinstance(node, ShowCollectionStmt):
            return self._execute_show_collection(node)
        if isinstance(node, ScrollStmt):
            return self._execute_scroll(node)
        if isinstance(node, SelectStmt):
            return self._execute_select(node)
        if isinstance(node, SearchStmt):
            return self._execute_search(node)
        if isinstance(node, RecommendStmt):
            return self._execute_recommend(node)
        if isinstance(node, DeleteStmt):
            return self._execute_delete(node)
        if isinstance(node, UpdateVectorStmt):
            return self._execute_update_vector(node)
        if isinstance(node, UpdatePayloadStmt):
            return self._execute_update_payload(node)
        raise QQLRuntimeError(f"Unknown AST node type: {type(node)}")

    # ── Statement executors ───────────────────────────────────────────────

    @staticmethod
    def _is_grpc_not_found_error(error: BaseException) -> bool:
        """Return True if *error* is a gRPC NOT_FOUND status."""
        from grpc import RpcError, StatusCode
        return isinstance(error, RpcError) and error.code() == StatusCode.NOT_FOUND

    def _fetch_collection_info(self, name: str):
        """Fetch full CollectionInfo for *name* in a single API call.

        Returns the CollectionInfo object when the collection exists, or
        ``None`` when the collection is not found (HTTP 404 or gRPC NOT_FOUND).
        Any other Qdrant error is re-raised as :class:`QQLRuntimeError`.
        """
        try:
            return self._client.get_collection(name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return None
            raise QQLRuntimeError(
                f"Qdrant error fetching collection '{name}': {e}"
            ) from e
        except ValueError as e:
            if f"Collection {name} not found" in str(e):
                return None
            raise QQLRuntimeError(
                f"Qdrant error fetching collection '{name}': {e}"
            ) from e
        except Exception as e:
            if self._is_grpc_not_found_error(e):
                return None
            raise QQLRuntimeError(
                f"Qdrant error fetching collection '{name}': {e}"
            ) from e

    def _topology_from_collection_info(self, info: Any) -> CollectionTopology:
        """Parse a CollectionInfo object into a :class:`CollectionTopology`.

        Separates API access (handled by :meth:`_fetch_collection_info`) from
        topology parsing so each concern can be tested independently.
        """
        params = info.config.params
        vectors = params.vectors  # type: ignore[union-attr]
        sparse_vectors = params.sparse_vectors or {}

        if isinstance(vectors, dict):
            dense_names = tuple(vectors.keys())
            dense_sizes: tuple[tuple[str, int], ...] = tuple(
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
            # Single unnamed dense vector
            dense_names = ()
            unnamed_size = getattr(vectors, "size", None)
            dense_sizes = (("", unnamed_size),) if unnamed_size is not None else ()
            has_unnamed_dense = True
            is_named_dense = False

        sparse_names = (
            tuple(sparse_vectors.keys()) if isinstance(sparse_vectors, dict) else ()
        )
        return CollectionTopology(
            exists=True,
            is_named_dense=is_named_dense,
            has_unnamed_dense=has_unnamed_dense,
            dense_names=dense_names,
            sparse_names=sparse_names,
            dense_sizes=dense_sizes,
        )

    def _resolve_topology(self, name: str) -> CollectionTopology:
        """Return the topology for *name* using exactly one Qdrant API call.

        Calls :meth:`_fetch_collection_info` once.  A 404 response is treated
        as ``exists=False``; any other error is propagated.
        """
        info = self._fetch_collection_info(name)
        if info is None:
            return CollectionTopology(exists=False, is_named_dense=False)
        return self._topology_from_collection_info(info)

    def _default_dense_vector_name(self) -> str:
        return self._config.default_dense_vector_name

    def _default_sparse_vector_name(self) -> str:
        return self._config.default_sparse_vector_name

    def _execute_insert(self, node: InsertStmt) -> ExecutionResult:
        if "text" not in node.values:
            raise QQLRuntimeError("INSERT requires a 'text' field in VALUES")

        topology = self._resolve_topology(node.collection)
        use_hybrid = node.hybrid or (topology.exists and topology.is_hybrid)

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

            dense_name = node.dense_vector or self._default_dense_vector_name()
            sparse_name = node.sparse_vector or self._default_sparse_vector_name()

            if topology.exists:
                resolved_dense = topology.dense_using(node.dense_vector)
                if resolved_dense is None:
                    raise QQLRuntimeError(
                        "Hybrid collections must use named dense vectors"
                    )
                dense_name = resolved_dense
                sparse_name = topology.sparse_using(node.sparse_vector)
            else:
                self._create_collection_and_wait(
                    collection_name=node.collection,
                    vectors_config={
                        dense_name: VectorParams(
                            size=len(dense_vector), distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        sparse_name: SparseVectorParams(modifier=Modifier.IDF)
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
                            vector={dense_name: dense_vector, sparse_name: sparse_vector},
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

        self._ensure_collection(
            node.collection, len(vector), topology, node.dense_vector
        )
        point_vector = self._build_dense_point_vector(topology, vector, node.dense_vector)

        point_id, payload = self._extract_point_id_and_payload(node.values)

        try:
            self._client.upsert(
                collection_name=node.collection,
                wait=True,
                points=[PointStruct(id=point_id, vector=point_vector, payload=payload)],
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

        topology = self._resolve_topology(node.collection)
        use_hybrid = node.hybrid or (topology.exists and topology.is_hybrid)

        # ── Hybrid bulk INSERT: dense + sparse vectors ─────────────────────
        if use_hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_embedder = Embedder(dense_model)
            sparse_embedder = SparseEmbedder(sparse_model_name)
            dense_name = node.dense_vector or self._default_dense_vector_name()
            sparse_name = node.sparse_vector or self._default_sparse_vector_name()
            if topology.exists:
                resolved_dense = topology.dense_using(node.dense_vector)
                if resolved_dense is None:
                    raise QQLRuntimeError(
                        "Hybrid collections must use named dense vectors"
                    )
                dense_name = resolved_dense
                sparse_name = topology.sparse_using(node.sparse_vector)

            first_dense_vector: list[float] | None = None
            points: list[PointStruct] = []
            for vals in node.values_list:
                point_id, payload = self._extract_point_id_and_payload(vals)
                dense_vector = dense_embedder.embed(vals["text"])
                if first_dense_vector is None:
                    first_dense_vector = dense_vector
                sparse_obj = sparse_embedder.embed(vals["text"])
                sparse_vector = SparseVector(
                    indices=sparse_obj["indices"], values=sparse_obj["values"]
                )
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={dense_name: dense_vector, sparse_name: sparse_vector},
                        payload=payload,
                    )
                )

            if not topology.exists:
                assert first_dense_vector is not None
                self._create_collection_and_wait(
                    collection_name=node.collection,
                    vectors_config={
                        dense_name: VectorParams(size=len(first_dense_vector), distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        sparse_name: SparseVectorParams(modifier=Modifier.IDF)
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

        first_vector: list[float] | None = None
        points = []
        for vals in node.values_list:
            vector = embedder.embed(vals["text"])
            if first_vector is None:
                first_vector = vector
            point_id, payload = self._extract_point_id_and_payload(vals)
            point_vector = self._build_dense_point_vector(
                topology, vector, node.dense_vector
            )
            points.append(
                PointStruct(id=point_id, vector=point_vector, payload=payload)
            )

        assert first_vector is not None
        self._ensure_collection(
            node.collection, len(first_vector), topology, node.dense_vector
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
            message=f"Inserted {len(points)} points",
        )

    def _execute_create(self, node: CreateCollectionStmt) -> ExecutionResult:
        if self._client.collection_exists(node.collection):
            return ExecutionResult(
                success=True,
                message=f"Collection '{node.collection}' already exists",
            )

        dense_model_name = node.model or self._config.default_model

        # Build optional quantization config (None when QUANTIZE clause absent)
        quant_config = (
            self._build_quantization_config(node.quantization)
            if node.quantization is not None
            else None
        )
        quant_label = (
            f", {node.quantization.type.value} quantization"
            if node.quantization is not None
            else ""
        )
        hnsw_config = self._build_hnsw_config(node.config)
        optimizers_config = self._build_optimizers_config(node.config)
        params_config = self._build_collection_params_create_kwargs(node.config)
        config_label = self._describe_collection_config(node.config)
        vector_on_disk = (
            node.config.vectors.on_disk
            if node.config is not None and node.config.vectors is not None
            else None
        )

        # ── Hybrid collection: named dense + sparse vectors ────────────────
        if node.hybrid:
            embedder = Embedder(dense_model_name)
            dims = embedder.dimensions
            dense_name = node.dense_vector or self._default_dense_vector_name()
            sparse_name = node.sparse_vector or self._default_sparse_vector_name()
            create_kwargs: dict[str, Any] = {
                "collection_name": node.collection,
                "vectors_config": {
                    dense_name: VectorParams(
                        size=dims,
                        distance=Distance.COSINE,
                        on_disk=vector_on_disk,
                    )
                },
                "sparse_vectors_config": {
                    sparse_name: SparseVectorParams(modifier=Modifier.IDF)
                },
            }
            if quant_config is not None:
                create_kwargs["quantization_config"] = quant_config
            if hnsw_config is not None:
                create_kwargs["hnsw_config"] = hnsw_config
            if optimizers_config is not None:
                create_kwargs["optimizers_config"] = optimizers_config
            create_kwargs.update(params_config)
            self._create_collection_and_wait(**create_kwargs)
            return ExecutionResult(
                success=True,
                message=(
                    f"Collection '{node.collection}' created "
                    f"(hybrid: {dims}-dim dense + BM25 sparse, cosine distance{quant_label}{config_label})"
                ),
            )

        # ── Standard dense-only collection ─────────────────────────────────
        embedder = Embedder(dense_model_name)
        dims = embedder.dimensions
        dense_name = node.dense_vector or self._default_dense_vector_name()
        create_kwargs = {
            "collection_name": node.collection,
            "vectors_config": {
                dense_name: VectorParams(
                    size=dims,
                    distance=Distance.COSINE,
                    on_disk=vector_on_disk,
                )
            },
        }
        if quant_config is not None:
            create_kwargs["quantization_config"] = quant_config
        if hnsw_config is not None:
            create_kwargs["hnsw_config"] = hnsw_config
        if optimizers_config is not None:
            create_kwargs["optimizers_config"] = optimizers_config
        create_kwargs.update(params_config)
        self._create_collection_and_wait(**create_kwargs)
        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' created ({dims}-dimensional vectors, cosine distance{quant_label}{config_label})",
        )

    def _execute_alter_collection(self, node: AlterCollectionStmt) -> ExecutionResult:
        topology = self._resolve_topology(node.collection)
        if not topology.exists:
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        update_kwargs: dict[str, Any] = {"collection_name": node.collection}
        vectors_config = self._build_vectors_config_diff(topology, node.config)
        hnsw_config = self._build_hnsw_config(node.config)
        optimizers_config = self._build_optimizers_config(node.config)
        collection_params = self._build_collection_params_diff(node.config)
        quantization_config = self._build_alter_quantization_config(node.quantization)

        if vectors_config is not None:
            update_kwargs["vectors_config"] = vectors_config
        if hnsw_config is not None:
            update_kwargs["hnsw_config"] = hnsw_config
        if optimizers_config is not None:
            update_kwargs["optimizers_config"] = optimizers_config
        if collection_params is not None:
            update_kwargs["collection_params"] = collection_params
        if quantization_config is not None:
            update_kwargs["quantization_config"] = quantization_config

        try:
            self._client.update_collection(**update_kwargs)
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during ALTER COLLECTION: {e}") from e

        return ExecutionResult(
            success=True,
            message=(
                f"Collection '{node.collection}' altered"
                f"{self._describe_collection_config(node.config)}"
                f"{self._describe_quantization_update(node.quantization)}"
            ),
        )

    def _execute_create_index(self, node: CreateIndexStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        schema_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
            "text": PayloadSchemaType.TEXT,
            "geo": PayloadSchemaType.GEO,
            "datetime": PayloadSchemaType.DATETIME,
            "uuid": PayloadSchemaType.UUID,
        }
        try:
            schema_map[node.schema]
        except KeyError as e:
            raise QQLRuntimeError(
                "Unknown index type '"
                f"{node.schema}'. Expected one of: keyword, integer, float, bool, text, geo, datetime, uuid"
            ) from e
        field_schema = self._build_payload_index_schema(node)

        try:
            self._client.create_payload_index(
                collection_name=node.collection,
                field_name=node.field_name,
                field_schema=field_schema,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during CREATE INDEX: {e}") from e

        option_label = f" with options {node.options}" if node.options else ""
        return ExecutionResult(
            success=True,
            message=(
                f"Created index on '{node.collection}.{node.field_name}' as '{node.schema}'{option_label}"
            ),
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

    def _execute_show_collection(self, node: ShowCollectionStmt) -> ExecutionResult:
        info = self._fetch_collection_info(node.collection)
        if info is None:
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        config = info.config
        params = config.params

        # ── Vector topology ────────────────────────────────────────────────
        vectors = params.vectors  # type: ignore[union-attr]
        sparse_vector_params = params.sparse_vectors or {}
        if isinstance(vectors, dict):
            vector_details = {}
            for vname, vconfig in vectors.items():
                vector_details[vname] = {
                    "size": vconfig.size,
                    "distance": str(vconfig.distance) if vconfig.distance else None,
                    "on_disk": vconfig.on_disk,
                }
        elif vectors is None:
            raise QQLRuntimeError(
                f"Collection '{node.collection}' has no vector configuration"
            )
        else:
            vector_details = {
                "": {
                    "size": vectors.size,
                    "distance": str(vectors.distance) if vectors.distance else None,
                    "on_disk": vectors.on_disk,
                }
            }
        topology = "hybrid" if sparse_vector_params else "dense"

        # ── Sparse vector config ───────────────────────────────────────────
        sparse_vectors = {}
        if sparse_vector_params:
            for sname, sconfig in sparse_vector_params.items():
                sparse_vectors[sname] = {
                    "modifier": str(sconfig.modifier) if sconfig.modifier else None,
                }

        # ── Quantization ───────────────────────────────────────────────────
        quant_config = config.quantization_config
        quantization = None
        if quant_config is not None:
            qtype = type(quant_config).__name__
            if hasattr(quant_config, "scalar"):
                quantization = "scalar"
            elif hasattr(quant_config, "binary"):
                quantization = "binary"
            elif hasattr(quant_config, "product"):
                quantization = "product"
            elif hasattr(quant_config, "turbo"):
                quantization = "turbo"
            else:
                quantization = qtype

        # ── HNSW config ────────────────────────────────────────────────────
        hnsw = {
            "m": config.hnsw_config.m,
            "ef_construct": config.hnsw_config.ef_construct,
        }
        if config.hnsw_config.full_scan_threshold is not None:
            hnsw["full_scan_threshold"] = config.hnsw_config.full_scan_threshold
        if config.hnsw_config.max_indexing_threads is not None:
            hnsw["max_indexing_threads"] = config.hnsw_config.max_indexing_threads
        if config.hnsw_config.on_disk is not None:
            hnsw["on_disk"] = config.hnsw_config.on_disk
        if config.hnsw_config.payload_m is not None:
            hnsw["payload_m"] = config.hnsw_config.payload_m
        if config.hnsw_config.inline_storage is not None:
            hnsw["inline_storage"] = config.hnsw_config.inline_storage

        # ── Payload schema / indexes ───────────────────────────────────────
        payload_indexes = {}
        for field_name, idx_info in (info.payload_schema or {}).items():
            payload_indexes[field_name] = self._serialize_payload_index_info(idx_info)

        # ── Sharding / replication ─────────────────────────────────────────
        sharding = {
            "shard_number": params.shard_number,
            "replication_factor": params.replication_factor,
            "write_consistency_factor": params.write_consistency_factor,
            "read_fan_out_factor": params.read_fan_out_factor,
            "read_fan_out_delay_ms": params.read_fan_out_delay_ms,
            "on_disk_payload": params.on_disk_payload,
        }

        data = {
            "name": node.collection,
            "status": str(info.status),
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "topology": topology,
            "vectors": vector_details,
            "sparse_vectors": sparse_vectors or None,
            "quantization": quantization,
            "hnsw_config": hnsw,
            "payload_schema": payload_indexes or None,
            "sharding": sharding,
        }

        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' diagnostics",
            data=data,
        )

    def _execute_scroll(self, node: ScrollStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        scroll_filter: Filter | None = None
        if node.query_filter is not None:
            scroll_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        try:
            records, next_offset = self._client.scroll(
                collection_name=node.collection,
                scroll_filter=scroll_filter,
                limit=node.limit,
                offset=node.after,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during SCROLL: {e}") from e

        points = [
            {"id": str(rec.id), "payload": rec.payload or {}}
            for rec in records
        ]
        return ExecutionResult(
            success=True,
            message=f"Scrolled {len(points)} point(s) from '{node.collection}'",
            data={"points": points, "next_offset": next_offset},
        )

    def _execute_select(self, node: SelectStmt) -> ExecutionResult:
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        try:
            records = self._client.retrieve(
                collection_name=node.collection,
                ids=[node.point_id],
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during SELECT: {e}") from e

        if not records:
            return ExecutionResult(
                success=True,
                message=f"Point '{node.point_id}' not found in '{node.collection}'",
            )

        record = records[0]
        return ExecutionResult(
            success=True,
            message=f"Retrieved point '{node.point_id}' from '{node.collection}'",
            data={"id": str(record.id), "payload": record.payload or {}},
        )

    def _execute_search(self, node: SearchStmt) -> ExecutionResult:
        topology = self._resolve_topology(node.collection)
        if not topology.exists:
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        # Build WHERE filter (shared by both hybrid and dense-only paths)
        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        search_params = self._build_search_params(node.with_clause)
        self._validate_search_mmr_usage(node)

        # When reranking is requested, fetch more candidates so the reranker has
        # enough material to reorder; only `node.limit` results are returned.
        fetch_limit = node.limit * _RERANK_FETCH_MULTIPLIER if node.rerank else node.limit

        lookup_from: LookupLocation | None = None
        if node.lookup_from is not None:
            lookup_from = LookupLocation(
                collection=node.lookup_from[0],
                vector=node.lookup_from[1],
            )

        # ── GROUP BY SEARCH: delegate to query_points_groups() ─────────────
        if node.group_by is not None:
            return self._execute_search_groups(
                node, qdrant_filter, search_params, topology
            )

        # ── Hybrid SEARCH: prefetch dense+sparse, fuse with the requested strategy ──
        if node.hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_vector, sparse_vector = self._build_hybrid_vectors(
                node.query_text, dense_model, sparse_model_name
            )

            try:
                response = self._client.query_points(
                    collection_name=node.collection,
                    prefetch=[
                        Prefetch(
                            query=self._build_dense_query(dense_vector, node.with_clause),
                            using=topology.dense_using(node.dense_vector),
                            limit=node.limit * _HYBRID_PREFETCH_MULTIPLIER,
                            params=search_params,
                        ),
                        Prefetch(
                            query=sparse_vector,
                            using=topology.sparse_using(node.sparse_vector),
                            limit=node.limit * _HYBRID_PREFETCH_MULTIPLIER,
                            params=search_params,
                        ),
                    ],
                    query=FusionQuery(fusion=self._resolve_hybrid_fusion(node.fusion)),
                    limit=fetch_limit,
                    offset=node.offset or None,
                    query_filter=qdrant_filter,
                    score_threshold=node.score_threshold,
                    lookup_from=lookup_from,
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

        # ── Sparse-only SEARCH: query the selected sparse vector directly ───
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
                    using=topology.sparse_using(node.sparse_vector),
                    limit=fetch_limit,
                    offset=node.offset or None,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    score_threshold=node.score_threshold,
                    lookup_from=lookup_from,
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
            query_using = topology.dense_using(node.dense_vector)
            response = self._client.query_points(
                collection_name=node.collection,
                query=self._build_dense_query(vector, node.with_clause),
                using=query_using,
                limit=fetch_limit,
                offset=node.offset or None,
                query_filter=qdrant_filter,
                search_params=search_params,
                score_threshold=node.score_threshold,
                lookup_from=lookup_from,
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

    def _build_hybrid_vectors(
        self,
        query_text: str,
        dense_model: str,
        sparse_model_name: str,
    ) -> tuple[list[float], SparseVector]:
        """Embed *query_text* with both dense and sparse models.

        Returns ``(dense_vector, sparse_vector)`` — a plain Python list for
        dense and a :class:`SparseVector` for sparse.  Extracted to eliminate
        duplication between the flat-hybrid and grouped-hybrid paths.
        """
        dense_vector: list[float] = Embedder(dense_model).embed(query_text)
        sparse_obj = SparseEmbedder(sparse_model_name).query_embed(query_text)
        sparse_vector = SparseVector(
            indices=sparse_obj["indices"],
            values=sparse_obj["values"],
        )
        return dense_vector, sparse_vector

    def _resolve_hybrid_fusion(self, fusion: str | None) -> Fusion:
        if fusion is None or fusion == "rrf":
            return Fusion.RRF
        if fusion == "dbsf":
            return Fusion.DBSF
        raise QQLRuntimeError(
            f"Unsupported hybrid fusion '{fusion}'; expected 'rrf' or 'dbsf'"
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
        if self._has_mmr(node.with_clause):
            raise QQLRuntimeError("MMR is supported only for SEARCH statements")

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

    def _build_payload_index_schema(self, node: CreateIndexStmt) -> Any:
        options = node.options or {}
        if node.schema == "keyword":
            self._validate_index_option_keys(
                node.schema,
                options,
                {"is_tenant", "on_disk", "enable_hnsw"},
            )
            if not options:
                return PayloadSchemaType.KEYWORD
            return KeywordIndexParams(
                type=KeywordIndexType.KEYWORD,
                is_tenant=self._index_bool_option(options, "is_tenant"),
                on_disk=self._index_bool_option(options, "on_disk"),
                enable_hnsw=self._index_bool_option(options, "enable_hnsw"),
            )

        if node.schema == "uuid":
            self._validate_index_option_keys(
                node.schema,
                options,
                {"is_tenant", "on_disk", "enable_hnsw"},
            )
            if not options:
                return PayloadSchemaType.UUID
            return UuidIndexParams(
                type=UuidIndexType.UUID,
                is_tenant=self._index_bool_option(options, "is_tenant"),
                on_disk=self._index_bool_option(options, "on_disk"),
                enable_hnsw=self._index_bool_option(options, "enable_hnsw"),
            )

        if node.schema == "text":
            self._validate_index_option_keys(
                node.schema,
                options,
                {
                    "tokenizer",
                    "min_token_len",
                    "max_token_len",
                    "lowercase",
                    "ascii_folding",
                    "phrase_matching",
                    "stopwords",
                    "on_disk",
                    "enable_hnsw",
                },
            )
            if not options:
                return PayloadSchemaType.TEXT
            min_token_len = self._index_int_option(options, "min_token_len")
            max_token_len = self._index_int_option(options, "max_token_len")
            if (
                min_token_len is not None
                and max_token_len is not None
                and min_token_len > max_token_len
            ):
                raise QQLRuntimeError(
                    "CREATE INDEX text option min_token_len cannot be greater than max_token_len"
                )
            return TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=self._text_tokenizer_option(options),
                min_token_len=min_token_len,
                max_token_len=max_token_len,
                lowercase=self._index_bool_option(options, "lowercase"),
                ascii_folding=self._index_bool_option(options, "ascii_folding"),
                phrase_matching=self._index_bool_option(options, "phrase_matching"),
                stopwords=self._text_stopwords_option(options),
                on_disk=self._index_bool_option(options, "on_disk"),
                enable_hnsw=self._index_bool_option(options, "enable_hnsw"),
            )

        if options:
            raise QQLRuntimeError(
                f"CREATE INDEX type '{node.schema}' does not support advanced options yet"
            )

        schema_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
            "text": PayloadSchemaType.TEXT,
            "geo": PayloadSchemaType.GEO,
            "datetime": PayloadSchemaType.DATETIME,
            "uuid": PayloadSchemaType.UUID,
        }
        return schema_map[node.schema]

    def _validate_index_option_keys(
        self,
        schema: str,
        options: dict[str, Any],
        allowed: set[str],
    ) -> None:
        unknown_keys = set(options) - allowed
        if unknown_keys:
            allowed_list = ", ".join(sorted(allowed))
            raise QQLRuntimeError(
                f"Unknown CREATE INDEX option '{sorted(unknown_keys)[0]}' for type '{schema}'. "
                f"Expected one of: {allowed_list}"
            )

    def _index_bool_option(self, options: dict[str, Any], key: str) -> bool | None:
        value = options.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise QQLRuntimeError(f"CREATE INDEX option '{key}' must be a boolean")
        return value

    def _index_int_option(self, options: dict[str, Any], key: str) -> int | None:
        value = options.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise QQLRuntimeError(
                f"CREATE INDEX option '{key}' must be a positive integer"
            )
        return value

    def _text_tokenizer_option(self, options: dict[str, Any]) -> TokenizerType | None:
        value = options.get("tokenizer")
        if value is None:
            return None
        if not isinstance(value, str):
            raise QQLRuntimeError("CREATE INDEX option 'tokenizer' must be a string")
        tokenizer_map = {
            "prefix": TokenizerType.PREFIX,
            "whitespace": TokenizerType.WHITESPACE,
            "word": TokenizerType.WORD,
            "multilingual": TokenizerType.MULTILINGUAL,
        }
        try:
            return tokenizer_map[value.lower()]
        except KeyError as e:
            raise QQLRuntimeError(
                "CREATE INDEX option 'tokenizer' must be one of: "
                "prefix, whitespace, word, multilingual"
            ) from e

    def _text_stopwords_option(
        self, options: dict[str, Any]
    ) -> Language | StopwordsSet | None:
        value = options.get("stopwords")
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return Language(value.lower())
            except ValueError as e:
                raise QQLRuntimeError(
                    "CREATE INDEX option 'stopwords' must be a known language name or a list of strings"
                ) from e
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return StopwordsSet(custom=value)
        raise QQLRuntimeError(
            "CREATE INDEX option 'stopwords' must be a string language name or a list of strings"
        )

    def _serialize_payload_index_info(self, idx_info: Any) -> dict[str, Any]:
        params = idx_info.params
        data = {"type": str(idx_info.data_type)}
        if params is None or not hasattr(params, "model_dump"):
            return data
        details: dict[str, Any] = {}
        for key, value in params.model_dump(exclude_none=True).items():
            if key == "type":
                continue
            details[key] = self._serialize_payload_index_value(value)
        if details:
            data["params"] = details
        return data

    def _serialize_payload_index_value(self, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, dict):
            return {
                key: self._serialize_payload_index_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._serialize_payload_index_value(item) for item in value]
        return value

    def _build_search_params(self, with_clause: SearchWith | None) -> SearchParams | None:
        if with_clause is None:
            return None
        quantization = None
        if with_clause.quantization is not None:
            quantization = QuantizationSearchParams(
                ignore=with_clause.quantization.ignore,
                rescore=with_clause.quantization.rescore,
                oversampling=with_clause.quantization.oversampling,
            )
        return SearchParams(
            hnsw_ef=with_clause.hnsw_ef,
            exact=with_clause.exact,
            quantization=quantization,
            indexed_only=True if with_clause.indexed_only else None,
            acorn=AcornSearchParams(enable=True) if with_clause.acorn else None,
        )

    def _build_hnsw_config(self, config: CollectionConfig | None) -> HnswConfigDiff | None:
        if config is None or config.hnsw is None:
            return None
        hnsw = config.hnsw
        return HnswConfigDiff(
            m=hnsw.m,
            ef_construct=hnsw.ef_construct,
            full_scan_threshold=hnsw.full_scan_threshold,
            max_indexing_threads=hnsw.max_indexing_threads,
            on_disk=hnsw.on_disk,
            payload_m=hnsw.payload_m,
            inline_storage=hnsw.inline_storage,
        )

    def _build_optimizers_config(
        self,
        config: CollectionConfig | None,
    ) -> OptimizersConfigDiff | None:
        if config is None or config.optimizers is None:
            return None
        optimizers = config.optimizers
        max_optimization_threads = optimizers.max_optimization_threads
        if max_optimization_threads == "auto":
            max_optimization_threads = MaxOptimizationThreadsSetting.AUTO
        return OptimizersConfigDiff(
            deleted_threshold=optimizers.deleted_threshold,
            vacuum_min_vector_number=optimizers.vacuum_min_vector_number,
            default_segment_number=optimizers.default_segment_number,
            max_segment_size=optimizers.max_segment_size,
            memmap_threshold=optimizers.memmap_threshold,
            indexing_threshold=optimizers.indexing_threshold,
            flush_interval_sec=optimizers.flush_interval_sec,
            max_optimization_threads=max_optimization_threads,
            prevent_unoptimized=optimizers.prevent_unoptimized,
        )

    def _build_collection_params_create_kwargs(
        self,
        config: CollectionConfig | None,
    ) -> dict[str, Any]:
        if config is None or config.params is None:
            return {}
        params = config.params
        create_kwargs: dict[str, Any] = {}
        if params.replication_factor is not None:
            create_kwargs["replication_factor"] = params.replication_factor
        if params.write_consistency_factor is not None:
            create_kwargs["write_consistency_factor"] = params.write_consistency_factor
        if params.on_disk_payload is not None:
            create_kwargs["on_disk_payload"] = params.on_disk_payload
        return create_kwargs

    def _build_collection_params_diff(
        self,
        config: CollectionConfig | None,
    ) -> CollectionParamsDiff | None:
        if config is None or config.params is None:
            return None
        params = config.params
        return CollectionParamsDiff(
            replication_factor=params.replication_factor,
            write_consistency_factor=params.write_consistency_factor,
            read_fan_out_factor=params.read_fan_out_factor,
            read_fan_out_delay_ms=params.read_fan_out_delay_ms,
            on_disk_payload=params.on_disk_payload,
        )

    def _build_vectors_config_diff(
        self,
        topology: CollectionTopology,
        config: CollectionConfig | None,
    ) -> dict[str, VectorParamsDiff] | None:
        if config is None or config.vectors is None:
            return None
        try:
            vector_name = topology.dense_config_key()
        except QQLRuntimeError as e:
            if "multiple dense vectors" in str(e):
                raise QQLRuntimeError(
                    "ALTER COLLECTION WITH VECTORS requires a collection with one dense vector"
                ) from e
            raise
        return {
            vector_name: VectorParamsDiff(on_disk=config.vectors.on_disk),
        }

    def _build_alter_quantization_config(
        self,
        quantization: QuantizationUpdate | None,
    ) -> (
        ScalarQuantization | BinaryQuantization | ProductQuantization | TurboQuantization | Disabled | None
    ):
        if quantization is None:
            return None
        if quantization.disabled:
            return Disabled.DISABLED
        if quantization.config is None:
            return None
        return self._build_quantization_config(quantization.config)

    def _describe_collection_config(self, config: CollectionConfig | None) -> str:
        if config is None:
            return ""
        labels: list[str] = []
        if config.vectors is not None and config.vectors.on_disk is not None:
            labels.append(f"vectors.on_disk={config.vectors.on_disk}")
        if config.hnsw is not None:
            hnsw = config.hnsw
            if hnsw.m is not None:
                labels.append(f"hnsw.m={hnsw.m}")
            if hnsw.ef_construct is not None:
                labels.append(f"hnsw.ef_construct={hnsw.ef_construct}")
            if hnsw.full_scan_threshold is not None:
                labels.append(f"hnsw.full_scan_threshold={hnsw.full_scan_threshold}")
            if hnsw.max_indexing_threads is not None:
                labels.append(f"hnsw.max_indexing_threads={hnsw.max_indexing_threads}")
            if hnsw.on_disk is not None:
                labels.append(f"hnsw.on_disk={hnsw.on_disk}")
            if hnsw.payload_m is not None:
                labels.append(f"hnsw.payload_m={hnsw.payload_m}")
            if hnsw.inline_storage is not None:
                labels.append(f"hnsw.inline_storage={hnsw.inline_storage}")
        if config.optimizers is not None:
            optimizers = config.optimizers
            for key in (
                "deleted_threshold",
                "vacuum_min_vector_number",
                "default_segment_number",
                "max_segment_size",
                "memmap_threshold",
                "indexing_threshold",
                "flush_interval_sec",
                "max_optimization_threads",
                "prevent_unoptimized",
            ):
                value = getattr(optimizers, key)
                if value is not None:
                    labels.append(f"optimizers.{key}={value}")
        if config.params is not None:
            params = config.params
            for key in (
                "replication_factor",
                "write_consistency_factor",
                "read_fan_out_factor",
                "read_fan_out_delay_ms",
                "on_disk_payload",
            ):
                value = getattr(params, key)
                if value is not None:
                    labels.append(f"params.{key}={value}")
        return f", {', '.join(labels)}" if labels else ""

    def _describe_quantization_update(
        self,
        quantization: QuantizationUpdate | None,
    ) -> str:
        if quantization is None:
            return ""
        if quantization.disabled:
            return ", quantization=disabled"
        if quantization.config is not None:
            return f", quantization={quantization.config.type.value}"
        return ""

    def _has_mmr(self, with_clause: SearchWith | None) -> bool:
        return with_clause is not None and (
            with_clause.mmr_diversity is not None or with_clause.mmr_candidates is not None
        )

    def _validate_search_mmr_usage(self, node: SearchStmt) -> None:
        if not self._has_mmr(node.with_clause):
            return
        if node.sparse_only:
            raise QQLRuntimeError("MMR is not supported with USING SPARSE yet")

    def _build_dense_query(
        self,
        vector: list[float],
        with_clause: SearchWith | None,
    ) -> list[float] | NearestQuery:
        if not self._has_mmr(with_clause):
            return vector
        return NearestQuery(
            nearest=vector,
            mmr=Mmr(
                diversity=with_clause.mmr_diversity,
                candidates_limit=with_clause.mmr_candidates,
            ),
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

    def _build_dense_point_vector(
        self,
        topology: CollectionTopology,
        vector: list[float],
        explicit_vector: str | None,
    ) -> list[float] | dict[str, list[float]]:
        if not topology.exists:
            return {explicit_vector or self._default_dense_vector_name(): vector}
        vector_name = topology.dense_payload_name(explicit_vector)
        if vector_name is None:
            return vector
        return {vector_name: vector}

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

        try:
            if node.query_filter is not None:
                self._client.delete(
                    collection_name=node.collection,
                    wait=True,
                    points_selector=self._wrap_as_filter(
                        self._build_qdrant_filter(node.query_filter)
                    ),
                )
                return ExecutionResult(
                    success=True,
                    message=f"Deleted points from '{node.collection}' by filter",
                )

            from qdrant_client.models import PointIdsList

            if node.point_id is None:
                raise QQLRuntimeError("DELETE requires either a point id or a filter")

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

    def _execute_search_groups(
        self,
        node: SearchStmt,
        qdrant_filter: Filter | None,
        search_params: SearchParams | None,
        topology: CollectionTopology,
    ) -> ExecutionResult:
        """Execute SEARCH ... GROUP BY using query_points_groups()."""
        
        lookup_from: LookupLocation | None = None
        if node.lookup_from is not None:
            lookup_from = LookupLocation(
                collection=node.lookup_from[0],
                vector=node.lookup_from[1],
            )
            
        try:
            if node.hybrid:
                dense_model = node.model or self._config.default_model
                sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
                dense_vector, sparse_vector = self._build_hybrid_vectors(
                    node.query_text, dense_model, sparse_model_name
                )
                response = self._client.query_points_groups(
                    collection_name=node.collection,
                    group_by=node.group_by,
                    prefetch=[
                        Prefetch(
                            query=self._build_dense_query(dense_vector, node.with_clause),
                            using=topology.dense_using(node.dense_vector),
                            limit=node.limit * _HYBRID_PREFETCH_MULTIPLIER,
                            params=search_params,
                        ),
                        Prefetch(
                            query=sparse_vector,
                            using=topology.sparse_using(node.sparse_vector),
                            limit=node.limit * _HYBRID_PREFETCH_MULTIPLIER,
                            params=search_params,
                        ),
                    ],
                    query=FusionQuery(fusion=self._resolve_hybrid_fusion(node.fusion)),
                    limit=node.limit,
                    group_size=node.group_size,
                    query_filter=qdrant_filter,
                    score_threshold=node.score_threshold,
                    lookup_from=lookup_from,
                )
                label = "hybrid, grouped"
            elif node.sparse_only:
                sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
                sparse_obj = SparseEmbedder(sparse_model_name).query_embed(node.query_text)
                sparse_vector = SparseVector(
                    indices=sparse_obj["indices"],
                    values=sparse_obj["values"],
                )
                response = self._client.query_points_groups(
                    collection_name=node.collection,
                    group_by=node.group_by,
                    query=sparse_vector,
                    using=topology.sparse_using(node.sparse_vector),
                    limit=node.limit,
                    group_size=node.group_size,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    score_threshold=node.score_threshold,
                    lookup_from=lookup_from,
                )
                label = "sparse, grouped"
            else:
                model_name = node.model or self._config.default_model
                vector = Embedder(model_name).embed(node.query_text)
                query_using = topology.dense_using(node.dense_vector)
                response = self._client.query_points_groups(
                    collection_name=node.collection,
                    group_by=node.group_by,
                    query=self._build_dense_query(vector, node.with_clause),
                    using=query_using,
                    limit=node.limit,
                    group_size=node.group_size,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    score_threshold=node.score_threshold,
                    lookup_from=lookup_from,
                )
                label = "grouped"
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during GROUP BY SEARCH: {e}") from e

        groups = [
            {
                "group_id": str(g.id),
                "hits": [
                    {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
                    for h in g.hits
                ],
            }
            for g in response.groups
        ]
        return ExecutionResult(
            success=True,
            message=f"Found {len(groups)} group(s) by '{node.group_by}' ({label})",
            data=groups,
        )

    def _execute_update_vector(self, node: UpdateVectorStmt) -> ExecutionResult:
        """Execute UPDATE ... SET VECTOR using update_vectors()."""
        topology = self._resolve_topology(node.collection)
        if not topology.exists:
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        vector_name = topology.dense_payload_name(node.vector_name)
        vector_struct: Any = (
            {vector_name: list(node.vector)} if vector_name else list(node.vector)
        )
        try:
            self._client.update_vectors(
                collection_name=node.collection,
                points=[PointVectors(id=node.point_id, vector=vector_struct)],
                wait=True,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during UPDATE VECTOR: {e}") from e
        return ExecutionResult(
            success=True,
            message=f"Updated vector for point [{node.point_id}] in '{node.collection}'",
            data=[],
        )

    def _execute_update_payload(self, node: UpdatePayloadStmt) -> ExecutionResult:
        """Execute UPDATE ... SET PAYLOAD using set_payload()."""
        if not self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        try:
            if node.query_filter is not None:
                qdrant_filter = self._wrap_as_filter(
                    self._build_qdrant_filter(node.query_filter)
                )
                self._client.set_payload(
                    collection_name=node.collection,
                    payload=node.payload,
                    points=qdrant_filter,
                    wait=True,
                )
                return ExecutionResult(
                    success=True,
                    message=f"Payload updated in '{node.collection}' (filter-based)",
                    data=[],
                )
            self._client.set_payload(
                collection_name=node.collection,
                payload=node.payload,
                points=[node.point_id],
                wait=True,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during UPDATE PAYLOAD: {e}") from e
        return ExecutionResult(
            success=True,
            message=f"Payload updated for point [{node.point_id}] in '{node.collection}'",
            data=[],
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
            non_nulls = [v for v in expr.values if v is not None]
            if len(non_nulls) == len(expr.values):
                return FieldCondition(
                    key=expr.field, match=MatchAny(any=non_nulls)
                )
            null_condition = IsNullCondition(is_null=PayloadField(key=expr.field))
            if not non_nulls:
                return null_condition
            return Filter(
                should=[
                    null_condition,
                    FieldCondition(key=expr.field, match=MatchAny(any=non_nulls)),
                ]
            )

        if isinstance(expr, NotInExpr):
            non_nulls = [v for v in expr.values if v is not None]
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

    def _build_quantization_config(
        self, qc: QuantizationConfig
    ) -> ScalarQuantization | BinaryQuantization | ProductQuantization | TurboQuantization:
        """Convert a parsed QuantizationConfig to a Qdrant SDK quantization object."""
        if qc.type == QuantizationType.SCALAR:
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=qc.quantile,      # None → SDK uses its own default (0.99)
                    always_ram=qc.always_ram,
                )
            )
        if qc.type == QuantizationType.BINARY:
            return BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=qc.always_ram)
            )
        if qc.type == QuantizationType.PRODUCT:
            return ProductQuantization(
                product=ProductQuantizationConfig(
                    compression=CompressionRatio.X4,
                    always_ram=qc.always_ram,
                )
            )
        if qc.type == QuantizationType.TURBO:
            _BITS_MAP: dict[float, TurboQuantBitSize] = {
                4.0: TurboQuantBitSize.BITS4,
                2.0: TurboQuantBitSize.BITS2,
                1.5: TurboQuantBitSize.BITS1_5,
                1.0: TurboQuantBitSize.BITS1,
            }
            if qc.turbo_bits is None:
                bits_enum = None           # user omitted BITS → preserve None, server applies default
            elif qc.turbo_bits in _BITS_MAP:
                bits_enum = _BITS_MAP[qc.turbo_bits]
            else:
                raise QQLRuntimeError(
                    f"Unsupported TURBO bit depth: {qc.turbo_bits}. "
                    f"Valid values: 1, 1.5, 2, 4"
                )
            return TurboQuantization(
                turbo=TurboQuantQuantizationConfig(
                    bits=bits_enum,
                    always_ram=qc.always_ram,
                )
            )
        raise QQLRuntimeError(f"Unknown quantization type: {qc.type}")

    def _ensure_collection(
        self,
        name: str,
        vector_size: int,
        topology: CollectionTopology,
        explicit_vector: str | None,
    ) -> None:
        """Create the collection if needed, or validate dimension compatibility.

        QQL-created dense collections use the configured dense vector name.
        Externally created unnamed collections still accept plain dense vectors.
        All validation is done against pre-fetched ``topology`` data; no extra
        Qdrant API calls are made.
        """
        if topology.exists:
            sizes = topology.dense_size_map()
            if topology.is_named_dense:
                # dense_using() raises QQLRuntimeError on bad/ambiguous names,
                # and always returns a non-None string in the named-dense branch.
                vector_name = topology.dense_using(explicit_vector)
                expected_size = sizes.get(vector_name)  # type: ignore[arg-type]
                if expected_size is not None and expected_size != vector_size:
                    raise QQLRuntimeError(
                        f"Vector dimension mismatch: collection '{name}' vector "
                        f"'{vector_name}' expects {expected_size} dims, but "
                        f"model produces {vector_size} dims. Specify a compatible "
                        "model with USING MODEL '<model>'."
                    )
            elif topology.has_unnamed_dense:
                expected_size = sizes.get("")
                if expected_size is not None and expected_size != vector_size:
                    raise QQLRuntimeError(
                        f"Vector dimension mismatch: collection '{name}' expects "
                        f"{expected_size} dims, but model produces {vector_size} dims. "
                        f"Specify a compatible model with USING MODEL '<model>'."
                    )
            else:
                raise QQLRuntimeError("Collection has no dense vector")
        else:
            self._create_collection_and_wait(
                collection_name=name,
                vectors_config={
                    explicit_vector or self._default_dense_vector_name(): VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    )
                },
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
