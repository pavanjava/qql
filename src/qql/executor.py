from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from .ast_nodes import (
    ASTNode,
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertStmt,
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
        # Create with default model dimensions
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

        try:
            response = self._client.query_points(
                collection_name=node.collection,
                query=vector,
                limit=node.limit,
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

    # ── Helpers ───────────────────────────────────────────────────────────

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
