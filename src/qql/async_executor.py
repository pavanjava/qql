from __future__ import annotations

import time
import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Filter,
    FusionQuery,
    LookupLocation,
    Modifier,
    PointStruct,
    PointVectors,
    Prefetch,
    RecommendInput,
    RecommendQuery,
    QueryRequest,
    SearchParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    PayloadSchemaType,
)

from .ast_nodes import (
    ASTNode,
    AlterCollectionStmt,
    CreateCollectionStmt,
    CreateIndexStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertBulkStmt,
    InsertStmt,
    RecommendStmt,
    SelectStmt,
    ScrollStmt,
    SearchStmt,
    ShowCollectionStmt,
    ShowCollectionsStmt,
    UpdateVectorStmt,
    UpdatePayloadStmt,
    BatchBlockStmt,
)
from .config import QQLConfig
from .embedder import Embedder, SparseEmbedder
from .exceptions import QQLRuntimeError
from .executor import Executor, ExecutionResult, CollectionTopology
from .utils import (
    build_bulk_insert_from_group,
    build_dense_point_vector,
    build_dense_query,
    collection_topology_kwargs,
    exclude_ids_from_filter,
    extract_point_id_and_payload,
    group_batch_statements,
    has_mmr,
    inserted_point_results,
    parse_recommend_strategy,
    resolve_hybrid_fusion,
    validate_search_mmr_usage,
)

_RERANK_FETCH_MULTIPLIER = 4
_HYBRID_PREFETCH_MULTIPLIER = 4
_COLLECTION_VISIBILITY_TIMEOUT_SECONDS = 5.0
_COLLECTION_VISIBILITY_POLL_SECONDS = 0.05


class AsyncExecutor(Executor):
    """Asynchronous QQL execution engine for ``AsyncQdrantClient``.

    The async executor mirrors :class:`~qql.Executor` at the statement boundary:
    every AST node supported by the sync executor has an async execution path
    here. Pure parsing, validation, vector-shaping, filter-building, and result
    formatting helpers live in ``qql.utils`` or are inherited from
    :class:`~qql.Executor`; only Qdrant client calls and collection-creation
    coordination are implemented with ``async``/``await`` in this module.
    """

    def __init__(self, client: AsyncQdrantClient, config: QQLConfig) -> None:
        super().__init__(client=client, config=config)  # type: ignore[arg-type]
        self._client: AsyncQdrantClient = client
        self._creation_lock = asyncio.Lock()

    async def execute(self, node: ASTNode) -> ExecutionResult:
        if isinstance(node, InsertBulkStmt):
            return await self._execute_insert_bulk(node)
        if isinstance(node, InsertStmt):
            return await self._execute_insert(node)
        if isinstance(node, CreateCollectionStmt):
            return await self._execute_create(node)
        if isinstance(node, AlterCollectionStmt):
            return await self._execute_alter_collection(node)
        if isinstance(node, CreateIndexStmt):
            return await self._execute_create_index(node)
        if isinstance(node, DropCollectionStmt):
            return await self._execute_drop(node)
        if isinstance(node, ShowCollectionsStmt):
            return await self._execute_show(node)
        if isinstance(node, ShowCollectionStmt):
            return await self._execute_show_collection(node)
        if isinstance(node, ScrollStmt):
            return await self._execute_scroll(node)
        if isinstance(node, SelectStmt):
            return await self._execute_select(node)
        if isinstance(node, SearchStmt):
            return await self._execute_search(node)
        if isinstance(node, RecommendStmt):
            return await self._execute_recommend(node)
        if isinstance(node, DeleteStmt):
            return await self._execute_delete(node)
        if isinstance(node, UpdateVectorStmt):
            return await self._execute_update_vector(node)
        if isinstance(node, UpdatePayloadStmt):
            return await self._execute_update_payload(node)
        if isinstance(node, BatchBlockStmt):
            return await self._execute_batch_block(node)
        raise QQLRuntimeError(f"Unknown AST node type: {type(node)}")

    # ── Topology & Helper methods ─────────────────────────────────────────

    async def _resolve_topology(self, name: str) -> CollectionTopology:
        if not await self._client.collection_exists(name):
            return CollectionTopology(exists=False, is_named_dense=False)

        info = await self._client.get_collection(name)
        params = info.config.params
        vectors = params.vectors  # type: ignore[union-attr]
        sparse_vectors = params.sparse_vectors or {}
        return CollectionTopology(**collection_topology_kwargs(vectors, sparse_vectors))

    async def _ensure_collection(
        self,
        name: str,
        vector_size: int,
        topology: CollectionTopology,
        explicit_vector: str | None,
    ) -> CollectionTopology:
        if topology.exists:
            info = await self._client.get_collection(name)
            vectors = info.config.params.vectors  # type: ignore[union-attr]
            sparse_vectors = info.config.params.sparse_vectors or {}
            current_topology = CollectionTopology(
                **collection_topology_kwargs(vectors, sparse_vectors)
            )
            if isinstance(vectors, dict):
                vector_name = current_topology.dense_using(explicit_vector)
                if vector_name is None:
                    raise QQLRuntimeError("Collection has no dense vector")
                vector_config = vectors[vector_name]
                expected_size = getattr(vector_config, "size", None)
                if expected_size is not None and expected_size != vector_size:
                    raise QQLRuntimeError(
                        f"Vector dimension mismatch: collection '{name}' vector "
                        f"'{vector_name}' expects {expected_size} dims, but "
                        f"model produces {vector_size} dims. Specify a compatible "
                        "model with USING MODEL '<model>'."
                    )
            elif vectors is not None:
                if vectors.size != vector_size:
                    raise QQLRuntimeError(
                        f"Vector dimension mismatch: collection '{name}' expects "
                        f"{vectors.size} dims, but model produces {vector_size} dims. "
                        f"Specify a compatible model with USING MODEL '<model>'."
                    )
            else:
                raise QQLRuntimeError("Collection has no dense vector")
            return current_topology
        else:
            async with self._creation_lock:
                current_topology = await self._resolve_topology(name)
                if current_topology.exists:
                    return await self._ensure_collection(name, vector_size, current_topology, explicit_vector)

                await self._create_collection_and_wait(
                    collection_name=name,
                    vectors_config={
                        explicit_vector or self._default_dense_vector_name(): VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        )
                    },
                )
                return await self._resolve_topology(name)

    async def _create_collection_and_wait(self, **kwargs: Any) -> None:
        collection_name = kwargs["collection_name"]
        await self._client.create_collection(**kwargs)

        deadline = time.monotonic() + _COLLECTION_VISIBILITY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if await self._client.collection_exists(collection_name):
                return
            await asyncio.sleep(_COLLECTION_VISIBILITY_POLL_SECONDS)

        raise QQLRuntimeError(
            f"Collection '{collection_name}' was created but did not become visible in time"
        )

    async def _build_hybrid_vectors(
        self,
        query_text: str,
        dense_model: str,
        sparse_model_name: str,
    ) -> tuple[list[float], SparseVector]:
        dense_embedder = Embedder(dense_model)
        sparse_embedder = SparseEmbedder(sparse_model_name)

        dense_vector = dense_embedder.embed(query_text)
        sparse_obj = sparse_embedder.query_embed(query_text)
        sparse_vector = SparseVector(
            indices=sparse_obj["indices"],
            values=sparse_obj["values"],
        )
        return dense_vector, sparse_vector

    # ── Statement executors ───────────────────────────────────────────────

    async def _execute_insert(self, node: InsertStmt) -> ExecutionResult:
        if "text" not in node.values:
            raise QQLRuntimeError("INSERT requires a 'text' field in VALUES")

        topology = await self._resolve_topology(node.collection)
        use_hybrid = node.hybrid or (topology.exists and topology.is_hybrid)

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
                async with self._creation_lock:
                    current_topology = await self._resolve_topology(node.collection)
                    if not current_topology.exists:
                        await self._create_collection_and_wait(
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
                    else:
                        dense_name = current_topology.dense_using(node.dense_vector) or dense_name
                        sparse_name = current_topology.sparse_using(node.sparse_vector)

            point_id, payload = extract_point_id_and_payload(node.values)
            try:
                await self._client.upsert(
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

        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.values["text"])

        topology = await self._ensure_collection(
            node.collection, len(vector), topology, node.dense_vector
        )
        point_vector = build_dense_point_vector(
            topology,
            vector,
            node.dense_vector,
            self._default_dense_vector_name(),
        )

        point_id, payload = extract_point_id_and_payload(node.values)

        try:
            await self._client.upsert(
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

    async def _execute_insert_bulk(self, node: InsertBulkStmt) -> ExecutionResult:
        if not node.values_list:
            raise QQLRuntimeError("INSERT BULK VALUES list is empty")
        for i, vals in enumerate(node.values_list):
            if "text" not in vals:
                raise QQLRuntimeError(
                    f"INSERT BULK: item at index {i} is missing required 'text' field"
                )

        topology = await self._resolve_topology(node.collection)
        use_hybrid = node.hybrid or (topology.exists and topology.is_hybrid)

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

            dense_vectors = [
                dense_embedder.embed(vals["text"]) for vals in node.values_list
            ]
            sparse_objs = [sparse_embedder.embed(vals["text"]) for vals in node.values_list]

            first_dense_vector = dense_vectors[0] if dense_vectors else None
            if not topology.exists:
                assert first_dense_vector is not None
                async with self._creation_lock:
                    current_topology = await self._resolve_topology(node.collection)
                    if not current_topology.exists:
                        await self._create_collection_and_wait(
                            collection_name=node.collection,
                            vectors_config={
                                dense_name: VectorParams(size=len(first_dense_vector), distance=Distance.COSINE)
                            },
                            sparse_vectors_config={
                                sparse_name: SparseVectorParams(modifier=Modifier.IDF)
                            },
                        )
                    else:
                        dense_name = current_topology.dense_using(node.dense_vector) or dense_name
                        sparse_name = current_topology.sparse_using(node.sparse_vector)

            points: list[PointStruct] = []
            for idx, vals in enumerate(node.values_list):
                point_id, payload = extract_point_id_and_payload(vals)
                dense_vector = dense_vectors[idx]
                sparse_obj = sparse_objs[idx]
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

            try:
                await self._client.upsert(
                    collection_name=node.collection,
                    wait=True,
                    points=points,
                )
            except UnexpectedResponse as e:
                raise QQLRuntimeError(f"Qdrant error during INSERT BULK: {e}") from e

            return ExecutionResult(
                success=True,
                message=f"Inserted {len(points)} points (hybrid)",
                data={"ids": [p.id for p in points]},
            )

        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)

        vectors = [embedder.embed(vals["text"]) for vals in node.values_list]

        first_vector = vectors[0] if vectors else None
        assert first_vector is not None
        topology = await self._ensure_collection(
            node.collection, len(first_vector), topology, node.dense_vector
        )
        points = []
        for idx, vals in enumerate(node.values_list):
            vector = vectors[idx]
            point_id, payload = extract_point_id_and_payload(vals)
            point_vector = build_dense_point_vector(
                topology,
                vector,
                node.dense_vector,
                self._default_dense_vector_name(),
            )
            points.append(
                PointStruct(id=point_id, vector=point_vector, payload=payload)
            )

        try:
            await self._client.upsert(
                collection_name=node.collection,
                wait=True,
                points=points,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during INSERT BULK: {e}") from e

        return ExecutionResult(
            success=True,
            message=f"Inserted {len(points)} points",
            data={"ids": [p.id for p in points]},
        )

    async def _execute_create(self, node: CreateCollectionStmt) -> ExecutionResult:
        if await self._client.collection_exists(node.collection):
            return ExecutionResult(
                success=True,
                message=f"Collection '{node.collection}' already exists",
            )

        dense_model_name = node.model or self._config.default_model

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
            await self._create_collection_and_wait(**create_kwargs)
            return ExecutionResult(
                success=True,
                message=(
                    f"Collection '{node.collection}' created "
                    f"(hybrid: {dims}-dim dense + BM25 sparse, cosine distance{quant_label}{config_label})"
                ),
            )

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
        await self._create_collection_and_wait(**create_kwargs)
        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' created ({dims}-dimensional vectors, cosine distance{quant_label}{config_label})",
        )

    async def _execute_alter_collection(self, node: AlterCollectionStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        topology = await self._resolve_topology(node.collection)

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
            await self._client.update_collection(**update_kwargs)
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

    async def _execute_create_index(self, node: CreateIndexStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
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
            await self._client.create_payload_index(
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

    async def _execute_drop(self, node: DropCollectionStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        await self._client.delete_collection(node.collection)
        return ExecutionResult(
            success=True,
            message=f"Collection '{node.collection}' dropped",
        )

    async def _execute_show(self, node: ShowCollectionsStmt) -> ExecutionResult:
        response = await self._client.get_collections()
        names = [c.name for c in response.collections]
        return ExecutionResult(
            success=True,
            message=f"{len(names)} collection(s) found",
            data=names,
        )

    async def _execute_show_collection(self, node: ShowCollectionStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        info = await self._client.get_collection(node.collection)
        config = info.config
        params = config.params

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

        sparse_vectors = {}
        if sparse_vector_params:
            for sname, sconfig in sparse_vector_params.items():
                sparse_vectors[sname] = {
                    "modifier": str(sconfig.modifier) if sconfig.modifier else None,
                }

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

        payload_indexes = {}
        for field_name, idx_info in (info.payload_schema or {}).items():
            payload_indexes[field_name] = self._serialize_payload_index_info(idx_info)

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

    async def _execute_scroll(self, node: ScrollStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        scroll_filter: Filter | None = None
        if node.query_filter is not None:
            scroll_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        try:
            records, next_offset = await self._client.scroll(
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

    async def _execute_select(self, node: SelectStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        try:
            records = await self._client.retrieve(
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

    async def _execute_search(self, node: SearchStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        topology = await self._resolve_topology(node.collection)

        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )

        search_params = self._build_search_params(node.with_clause)
        validate_search_mmr_usage(node)

        fetch_limit = node.limit * _RERANK_FETCH_MULTIPLIER if node.rerank else node.limit

        lookup_from: LookupLocation | None = None
        if node.lookup_from is not None:
            lookup_from = LookupLocation(
                collection=node.lookup_from[0],
                vector=node.lookup_from[1],
            )

        if node.group_by is not None:
            return await self._execute_search_groups(
                node, qdrant_filter, search_params, topology
            )

        if node.hybrid:
            dense_model = node.model or self._config.default_model
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            dense_vector, sparse_vector = await self._build_hybrid_vectors(
                node.query_text, dense_model, sparse_model_name
            )

            try:
                response = await self._client.query_points(
                    collection_name=node.collection,
                    prefetch=[
                        Prefetch(
                            query=build_dense_query(dense_vector, node.with_clause),
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
                    query=FusionQuery(fusion=resolve_hybrid_fusion(node.fusion)),
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

        if node.sparse_only:
            sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
            sparse_embedder = SparseEmbedder(sparse_model_name)
            sparse_obj = sparse_embedder.query_embed(node.query_text)
            sparse_vector = SparseVector(
                indices=sparse_obj["indices"],
                values=sparse_obj["values"],
            )

            try:
                response = await self._client.query_points(
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

        model_name = node.model or self._config.default_model
        embedder = Embedder(model_name)
        vector = embedder.embed(node.query_text)

        try:
            query_using = topology.dense_using(node.dense_vector)
            response = await self._client.query_points(
                collection_name=node.collection,
                query=build_dense_query(vector, node.with_clause),
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

    async def _execute_search_groups(
        self,
        node: SearchStmt,
        qdrant_filter: Filter | None,
        search_params: SearchParams | None,
        topology: CollectionTopology,
    ) -> ExecutionResult:
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
                dense_vector, sparse_vector = await self._build_hybrid_vectors(
                    node.query_text, dense_model, sparse_model_name
                )
                response = await self._client.query_points_groups(
                    collection_name=node.collection,
                    group_by=node.group_by,
                    prefetch=[
                        Prefetch(
                            query=build_dense_query(dense_vector, node.with_clause),
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
                    query=FusionQuery(fusion=resolve_hybrid_fusion(node.fusion)),
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
                response = await self._client.query_points_groups(
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
                response = await self._client.query_points_groups(
                    collection_name=node.collection,
                    group_by=node.group_by,
                    query=build_dense_query(vector, node.with_clause),
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

    async def _execute_recommend(self, node: RecommendStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        qdrant_filter: Filter | None = None
        if node.query_filter is not None:
            qdrant_filter = self._wrap_as_filter(
                self._build_qdrant_filter(node.query_filter)
            )
        qdrant_filter = exclude_ids_from_filter(
            qdrant_filter,
            [*node.positive_ids, *node.negative_ids],
        )

        recommend_input = RecommendInput(
            positive=list(node.positive_ids),
            negative=list(node.negative_ids) or None,
            strategy=parse_recommend_strategy(node.strategy),
        )

        search_params = self._build_search_params(node.with_clause)
        if has_mmr(node.with_clause):
            raise QQLRuntimeError("MMR is supported only for SEARCH statements")

        lookup_from: LookupLocation | None = None
        if node.lookup_from is not None:
            lookup_from = LookupLocation(
                collection=node.lookup_from[0],
                vector=node.lookup_from[1],
            )

        try:
            response = await self._client.query_points(
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

    async def _execute_delete(self, node: DeleteStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")

        try:
            if node.query_filter is not None:
                await self._client.delete(
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

            await self._client.delete(
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

    async def _execute_update_vector(self, node: UpdateVectorStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        topology = await self._resolve_topology(node.collection)
        vector_name = topology.dense_payload_name(node.vector_name)
        vector_struct: Any = (
            {vector_name: list(node.vector)} if vector_name else list(node.vector)
        )
        try:
            await self._client.update_vectors(
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

    async def _execute_update_payload(self, node: UpdatePayloadStmt) -> ExecutionResult:
        if not await self._client.collection_exists(node.collection):
            raise QQLRuntimeError(f"Collection '{node.collection}' does not exist")
        try:
            if node.query_filter is not None:
                qdrant_filter = self._wrap_as_filter(
                    self._build_qdrant_filter(node.query_filter)
                )
                await self._client.set_payload(
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
            await self._client.set_payload(
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

    async def _execute_batch_block(self, node: BatchBlockStmt) -> ExecutionResult:
        if not node.statements:
            return ExecutionResult(success=True, message="Executed empty batch", data=[])

        all_results = []
        succeeded_count = 0

        for group in group_batch_statements(node.statements):
            if group.kind == 'query':
                res = await self._execute_query_batch(group.collection, group.statements)
                all_results.extend(res)
                succeeded_count += len([r for r in res if r.success])
            elif group.kind == 'insert':
                bulk_node = build_bulk_insert_from_group(
                    group.collection,
                    group.statements,
                )
                res = await self._execute_insert_bulk(bulk_node)
                insert_results = inserted_point_results(
                    res,
                    group.statements,
                    ExecutionResult,
                )
                all_results.extend(insert_results)
                succeeded_count += len([r for r in insert_results if r.success])
            else:
                for s in group.statements:
                    res = await self.execute(s)
                    all_results.append(res)
                    if res.success:
                        succeeded_count += 1

        total_stmts = len(node.statements)
        return ExecutionResult(
            success=succeeded_count == total_stmts,
            message=f"Batch executed {succeeded_count}/{total_stmts} statement(s) successfully",
            data=all_results,
        )

    async def _execute_query_batch(
        self,
        collection_name: str,
        nodes: list[SearchStmt | RecommendStmt],
    ) -> list[ExecutionResult]:
        if not await self._client.collection_exists(collection_name):
            raise QQLRuntimeError(f"Collection '{collection_name}' does not exist")

        topology = await self._resolve_topology(collection_name)
        requests = []

        for node in nodes:
            qdrant_filter = None
            if node.query_filter is not None:
                qdrant_filter = self._wrap_as_filter(
                    self._build_qdrant_filter(node.query_filter)
                )

            search_params = self._build_search_params(node.with_clause)

            lookup_from = None
            if node.lookup_from is not None:
                lookup_from = LookupLocation(
                    collection=node.lookup_from[0],
                    vector=node.lookup_from[1],
                )

            if isinstance(node, SearchStmt):
                validate_search_mmr_usage(node)
                fetch_limit = node.limit * _RERANK_FETCH_MULTIPLIER if node.rerank else node.limit

                if node.hybrid:
                    dense_model = node.model or self._config.default_model
                    sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
                    dense_vector, sparse_vector = await self._build_hybrid_vectors(
                        node.query_text, dense_model, sparse_model_name
                    )

                    req = QueryRequest(
                        prefetch=[
                            Prefetch(
                                query=build_dense_query(dense_vector, node.with_clause),
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
                        query=FusionQuery(fusion=resolve_hybrid_fusion(node.fusion)),
                        limit=fetch_limit,
                        offset=node.offset or None,
                        filter=qdrant_filter,
                        score_threshold=node.score_threshold,
                        lookup_from=lookup_from,
                        with_payload=True,
                        with_vector=False,
                    )
                elif node.sparse_only:
                    sparse_model_name = node.sparse_model or SparseEmbedder.DEFAULT_MODEL
                    sparse_embedder = SparseEmbedder(sparse_model_name)
                    sparse_obj = sparse_embedder.query_embed(node.query_text)
                    sparse_vector = SparseVector(
                        indices=sparse_obj["indices"],
                        values=sparse_obj["values"],
                    )

                    req = QueryRequest(
                        query=sparse_vector,
                        using=topology.sparse_using(node.sparse_vector),
                        limit=fetch_limit,
                        offset=node.offset or None,
                        filter=qdrant_filter,
                        params=search_params,
                        score_threshold=node.score_threshold,
                        lookup_from=lookup_from,
                        with_payload=True,
                        with_vector=False,
                    )
                else:
                    model_name = node.model or self._config.default_model
                    embedder = Embedder(model_name)
                    vector = embedder.embed(node.query_text)
                    query_using = topology.dense_using(node.dense_vector)

                    req = QueryRequest(
                        query=build_dense_query(vector, node.with_clause),
                        using=query_using,
                        limit=fetch_limit,
                        offset=node.offset or None,
                        filter=qdrant_filter,
                        params=search_params,
                        score_threshold=node.score_threshold,
                        lookup_from=lookup_from,
                        with_payload=True,
                        with_vector=False,
                    )
            else:
                qdrant_filter = exclude_ids_from_filter(
                    qdrant_filter,
                    [*node.positive_ids, *node.negative_ids],
                )
                recommend_input = RecommendInput(
                    positive=list(node.positive_ids),
                    negative=list(node.negative_ids) or None,
                    strategy=parse_recommend_strategy(node.strategy),
                )
                if has_mmr(node.with_clause):
                    raise QQLRuntimeError("MMR is supported only for SEARCH statements")

                req = QueryRequest(
                    query=RecommendQuery(recommend=recommend_input),
                    limit=node.limit,
                    offset=node.offset or None,
                    filter=qdrant_filter,
                    params=search_params,
                    score_threshold=node.score_threshold,
                    using=node.using,
                    lookup_from=lookup_from,
                    with_payload=True,
                    with_vector=False,
                )

            requests.append(req)

        try:
            responses = await self._client.query_batch_points(
                collection_name=collection_name,
                requests=requests,
            )
        except UnexpectedResponse as e:
            raise QQLRuntimeError(f"Qdrant error during Batch Query: {e}") from e

        execution_results = []
        for i, response in enumerate(responses):
            node = nodes[i]
            results = [
                {"id": str(h.id), "score": round(h.score, 4), "payload": h.payload}
                for h in response.points
            ]

            if isinstance(node, SearchStmt) and node.rerank:
                results = self._apply_reranking(node.query_text, results, node.limit, node.rerank_model)
                label = "hybrid, reranked" if node.hybrid else ("sparse, reranked" if node.sparse_only else "reranked")
                msg = f"Found {len(results)} result(s) ({label})"
            else:
                if isinstance(node, SearchStmt):
                    label = "hybrid" if node.hybrid else ("sparse" if node.sparse_only else "")
                    label_suffix = f" ({label})" if label else ""
                    msg = f"Found {len(results)} result(s){label_suffix}"
                else:
                    msg = f"Found {len(results)} recommendation(s)"

            execution_results.append(
                ExecutionResult(success=True, message=msg, data=results)
            )

        return execution_results
