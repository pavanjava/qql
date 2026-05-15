import pytest

from qql.ast_nodes import (
    CreateCollectionStmt,
    CreateIndexStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertBulkStmt,
    InsertStmt,
    QuantizationConfig,
    QuantizationSearchWith,
    QuantizationType,
    RecommendStmt,
    SelectStmt,
    ScrollStmt,
    SearchStmt,
    SearchWith,
    ShowCollectionStmt,
    ShowCollectionsStmt,
)
from qql.config import QQLConfig
from qql.exceptions import QQLRuntimeError
from qql.executor import Executor


FAKE_VECTOR = [0.1] * 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def cfg():
    return QQLConfig(url="http://localhost:6333", secret=None)


@pytest.fixture
def mock_client(mocker):
    client = mocker.MagicMock()
    client.collection_exists.return_value = False
    state = {"exists": False}

    def collection_exists(_name):
        return state["exists"] or bool(client.collection_exists.return_value)

    def create_collection(**_kwargs):
        state["exists"] = True

    client.collection_exists.side_effect = collection_exists
    client.create_collection.side_effect = create_collection
    return client


@pytest.fixture
def executor(mock_client, cfg):
    return Executor(mock_client, cfg)


@pytest.fixture(autouse=True)
def mock_embedder(mocker):
    mock_embed = mocker.MagicMock()
    mock_embed.embed.return_value = FAKE_VECTOR
    mock_embed.dimensions = 384
    mocker.patch("qql.executor.Embedder", return_value=mock_embed)
    return mock_embed


class TestInsert:
    def test_insert_creates_collection_when_missing(self, executor, mock_client):
        node = InsertStmt(collection="notes", values={"text": "hello"}, model=None)
        executor.execute(node)
        mock_client.create_collection.assert_called_once()

    def test_insert_skips_create_when_collection_exists(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        # Simulate same vector size
        mock_client.get_collection.return_value.config.params.vectors.size = 384
        node = InsertStmt(collection="notes", values={"text": "hello"}, model=None)
        executor.execute(node)
        mock_client.create_collection.assert_not_called()

    def test_insert_calls_upsert(self, executor, mock_client):
        node = InsertStmt(collection="notes", values={"text": "hello", "author": "alice"}, model=None)
        result = executor.execute(node)
        mock_client.upsert.assert_called_once()
        assert result.success is True
        assert "Inserted 1 point" in result.message

    def test_insert_result_contains_point_id(self, executor, mock_client):
        node = InsertStmt(collection="notes", values={"text": "hi"}, model=None)
        result = executor.execute(node)
        assert result.data["id"] is not None
        assert len(result.data["id"]) == 36  # UUID format

    def test_insert_uses_explicit_uuid_id_when_provided(self, executor, mock_client):
        node = InsertStmt(
            collection="notes",
            values={"id": "550e8400-e29b-41d4-a716-446655440000", "text": "hello"},
            model=None,
        )
        result = executor.execute(node)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.id == "550e8400-e29b-41d4-a716-446655440000"
        assert "id" not in point.payload
        assert result.data["id"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_insert_uses_explicit_integer_id_when_provided(self, executor, mock_client):
        node = InsertStmt(
            collection="notes",
            values={"id": 42, "text": "hello"},
            model=None,
        )
        executor.execute(node)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.id == 42

    def test_insert_rejects_non_scalar_id(self, executor):
        node = InsertStmt(
            collection="notes",
            values={"id": {"bad": "id"}, "text": "hello"},
            model=None,
        )
        with pytest.raises(QQLRuntimeError, match="unsigned integer or UUID string"):
            executor.execute(node)

    def test_insert_rejects_non_uuid_string_id(self, executor):
        node = InsertStmt(
            collection="notes",
            values={"id": "note-1", "text": "hello"},
            model=None,
        )
        with pytest.raises(QQLRuntimeError, match="unsigned integer or UUID string"):
            executor.execute(node)

    def test_insert_stores_text_in_payload(self, executor, mock_client):
        node = InsertStmt(collection="notes", values={"text": "hello"}, model=None)
        executor.execute(node)
        call_args = mock_client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0].payload["text"] == "hello"

    def test_insert_raises_when_text_missing(self, executor):
        node = InsertStmt(collection="notes", values={"author": "alice"}, model=None)
        with pytest.raises(QQLRuntimeError, match="'text' field"):
            executor.execute(node)

    def test_insert_raises_on_dimension_mismatch(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value.config.params.vectors.size = 768
        node = InsertStmt(collection="notes", values={"text": "hi"}, model=None)
        with pytest.raises(QQLRuntimeError, match="dimension mismatch"):
            executor.execute(node)


class TestInsertBulk:
    def test_bulk_insert_calls_upsert_once(self, executor, mock_client):
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "hello"}, {"text": "world"}),
            model=None,
        )
        executor.execute(node)
        mock_client.upsert.assert_called_once()

    def test_bulk_insert_upserts_correct_count(self, executor, mock_client):
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "a"}, {"text": "b"}, {"text": "c"}),
            model=None,
        )
        executor.execute(node)
        call_args = mock_client.upsert.call_args.kwargs
        assert len(call_args["points"]) == 3

    def test_bulk_insert_creates_collection_when_missing(self, executor, mock_client):
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "hello"},),
            model=None,
        )
        executor.execute(node)
        mock_client.create_collection.assert_called_once()

    def test_bulk_insert_skips_create_when_exists(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value.config.params.vectors.size = 384
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "hello"},),
            model=None,
        )
        executor.execute(node)
        mock_client.create_collection.assert_not_called()

    def test_bulk_insert_raises_on_missing_text(self, executor):
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "ok"}, {"author": "bob"}),
            model=None,
        )
        with pytest.raises(QQLRuntimeError, match="index 1"):
            executor.execute(node)

    def test_bulk_insert_empty_list_raises(self, executor):
        node = InsertBulkStmt(collection="col", values_list=(), model=None)
        with pytest.raises(QQLRuntimeError, match="empty"):
            executor.execute(node)

    def test_bulk_insert_result_message_contains_count(self, executor, mock_client):
        node = InsertBulkStmt(
            collection="col",
            values_list=({"text": "a"}, {"text": "b"}),
            model=None,
        )
        result = executor.execute(node)
        assert result.success is True
        assert "2" in result.message
        assert "points" in result.message

    def test_bulk_insert_preserves_explicit_ids(self, executor, mock_client):
        node = InsertBulkStmt(
            collection="col",
            values_list=(
                {"id": "550e8400-e29b-41d4-a716-446655440001", "text": "a"},
                {"id": 2, "text": "b"},
            ),
            model=None,
        )
        executor.execute(node)
        points = mock_client.upsert.call_args.kwargs["points"]
        assert [point.id for point in points] == ["550e8400-e29b-41d4-a716-446655440001", 2]
        assert all("id" not in point.payload for point in points)

    def test_single_insert_unaffected_by_bulk_dispatch(self, executor, mock_client):
        """Ensure single INSERT still routes correctly after bulk dispatch added."""
        node = InsertStmt(collection="notes", values={"text": "hello"}, model=None)
        result = executor.execute(node)
        assert result.success is True
        assert "Inserted 1 point" in result.message


class TestCreate:
    def test_create_new_collection(self, executor, mock_client):
        node = CreateCollectionStmt(collection="new_col")
        result = executor.execute(node)
        mock_client.create_collection.assert_called_once()
        assert result.success is True

    def test_create_collection_passes_payload_m(self, executor, mock_client):
        from qdrant_client.models import HnswConfigDiff

        node = CreateCollectionStmt(collection="new_col", payload_m=24)
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["hnsw_config"], HnswConfigDiff)
        assert kw["hnsw_config"].payload_m == 24

    def test_create_existing_collection_is_noop(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateCollectionStmt(collection="existing")
        result = executor.execute(node)
        mock_client.create_collection.assert_not_called()
        assert result.success is True
        assert "already exists" in result.message


class TestCreateIndex:
    def test_create_index_calls_qdrant(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateIndexStmt(collection="articles", field_name="category", schema="keyword")
        result = executor.execute(node)
        mock_client.create_payload_index.assert_called_once()
        assert result.success is True

    def test_create_index_supports_keyword_options(self, executor, mock_client):
        from qdrant_client.models import KeywordIndexParams

        mock_client.collection_exists.return_value = True
        node = CreateIndexStmt(
            collection="articles",
            field_name="tenant_id",
            schema="keyword",
            options={"is_tenant": True, "on_disk": True, "enable_hnsw": False},
        )
        executor.execute(node)
        field_schema = mock_client.create_payload_index.call_args.kwargs["field_schema"]
        assert isinstance(field_schema, KeywordIndexParams)
        assert field_schema.is_tenant is True
        assert field_schema.on_disk is True
        assert field_schema.enable_hnsw is False

    def test_create_index_supports_text_options(self, executor, mock_client):
        from qdrant_client.models import TextIndexParams, TokenizerType

        mock_client.collection_exists.return_value = True
        node = CreateIndexStmt(
            collection="articles",
            field_name="title",
            schema="text",
            options={
                "tokenizer": "word",
                "min_token_len": 2,
                "max_token_len": 20,
                "lowercase": True,
                "phrase_matching": True,
            },
        )
        executor.execute(node)
        field_schema = mock_client.create_payload_index.call_args.kwargs["field_schema"]
        assert isinstance(field_schema, TextIndexParams)
        assert field_schema.tokenizer == TokenizerType.WORD
        assert field_schema.min_token_len == 2
        assert field_schema.max_token_len == 20
        assert field_schema.lowercase is True
        assert field_schema.phrase_matching is True

    def test_create_index_supports_uuid_options(self, executor, mock_client):
        from qdrant_client.models import UuidIndexParams

        mock_client.collection_exists.return_value = True
        node = CreateIndexStmt(
            collection="articles",
            field_name="doc_id",
            schema="uuid",
            options={"on_disk": True},
        )
        executor.execute(node)
        field_schema = mock_client.create_payload_index.call_args.kwargs["field_schema"]
        assert isinstance(field_schema, UuidIndexParams)
        assert field_schema.on_disk is True

    def test_create_index_rejects_unknown_option(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateIndexStmt(
            collection="articles",
            field_name="tenant_id",
            schema="keyword",
            options={"tokenizer": "word"},
        )
        with pytest.raises(QQLRuntimeError, match="Unknown CREATE INDEX option"):
            executor.execute(node)

    def test_create_index_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = CreateIndexStmt(collection="ghost", field_name="category", schema="keyword")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestCreateWithModel:
    def test_create_with_model_passes_model_to_embedder(self, mock_client, cfg, mocker):
        mock_emb = mocker.MagicMock()
        mock_emb.dimensions = 768
        embedder_cls = mocker.patch("qql.executor.Embedder", return_value=mock_emb)
        executor = Executor(mock_client, cfg)
        node = CreateCollectionStmt(collection="col", model="BAAI/bge-base-en-v1.5")
        executor.execute(node)
        embedder_cls.assert_called_once_with("BAAI/bge-base-en-v1.5")

    def test_create_without_model_uses_default_model(self, mock_client, cfg, mocker):
        mock_emb = mocker.MagicMock()
        mock_emb.dimensions = 384
        embedder_cls = mocker.patch("qql.executor.Embedder", return_value=mock_emb)
        executor = Executor(mock_client, cfg)
        node = CreateCollectionStmt(collection="col")
        executor.execute(node)
        embedder_cls.assert_called_once_with(cfg.default_model)

    def test_create_hybrid_with_model_uses_named_vectors(self, mock_client, cfg, mocker):
        mock_emb = mocker.MagicMock()
        mock_emb.dimensions = 768
        embedder_cls = mocker.patch("qql.executor.Embedder", return_value=mock_emb)
        executor = Executor(mock_client, cfg)
        node = CreateCollectionStmt(collection="col", hybrid=True, model="BAAI/bge-base-en-v1.5")
        executor.execute(node)
        embedder_cls.assert_called_once_with("BAAI/bge-base-en-v1.5")
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["vectors_config"], dict)
        assert "dense" in kw["vectors_config"]
        assert "sparse_vectors_config" in kw

    def test_create_hybrid_without_model_uses_default(self, mock_client, cfg, mocker):
        mock_emb = mocker.MagicMock()
        mock_emb.dimensions = 384
        embedder_cls = mocker.patch("qql.executor.Embedder", return_value=mock_emb)
        executor = Executor(mock_client, cfg)
        node = CreateCollectionStmt(collection="col", hybrid=True)
        executor.execute(node)
        embedder_cls.assert_called_once_with(cfg.default_model)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["vectors_config"], dict)

    def test_create_dense_with_model_uses_scalar_vectors(self, mock_client, cfg, mocker):
        from qdrant_client.models import VectorParams
        mock_emb = mocker.MagicMock()
        mock_emb.dimensions = 768
        mocker.patch("qql.executor.Embedder", return_value=mock_emb)
        executor = Executor(mock_client, cfg)
        node = CreateCollectionStmt(collection="col", model="BAAI/bge-base-en-v1.5")
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["vectors_config"], VectorParams)
        assert "sparse_vectors_config" not in kw

    def test_create_existing_noop_with_model(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateCollectionStmt(collection="col", model="some/model")
        result = executor.execute(node)
        mock_client.create_collection.assert_not_called()
        assert result.success is True
        assert "already exists" in result.message


class TestDrop:
    def test_drop_existing_collection(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = DropCollectionStmt(collection="old_col")
        result = executor.execute(node)
        mock_client.delete_collection.assert_called_once_with("old_col")
        assert result.success is True

    def test_drop_nonexistent_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = DropCollectionStmt(collection="ghost")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestShow:
    def test_show_returns_collection_names(self, executor, mock_client, mocker):
        col1 = mocker.MagicMock()
        col1.name = "notes"
        col2 = mocker.MagicMock()
        col2.name = "docs"
        mock_client.get_collections.return_value.collections = [col1, col2]
        node = ShowCollectionsStmt()
        result = executor.execute(node)
        assert result.success is True
        assert "notes" in result.data
        assert "docs" in result.data


class TestShowCollection:
    def test_show_collection_returns_diagnostics(self, executor, mock_client, mocker):
        from qdrant_client.models import (
            CollectionStatus,
            Distance,
            VectorParams,
        )

        mock_client.collection_exists.return_value = True

        mock_info = mocker.MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.points_count = 42
        mock_info.indexed_vectors_count = 42
        mock_info.segments_count = 2

        mock_info.config.params.vectors = VectorParams(size=384, distance=Distance.COSINE)
        mock_info.config.params.shard_number = 1
        mock_info.config.params.replication_factor = 1
        mock_info.config.params.write_consistency_factor = 1
        mock_info.config.params.sparse_vectors = None
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 100
        mock_info.config.hnsw_config.full_scan_threshold = None
        mock_info.config.hnsw_config.max_indexing_threads = None
        mock_info.config.hnsw_config.on_disk = None
        mock_info.config.hnsw_config.payload_m = None
        mock_info.config.quantization_config = None
        mock_info.payload_schema = {}

        mock_client.get_collection.return_value = mock_info

        node = ShowCollectionStmt(collection="docs")
        result = executor.execute(node)

        assert result.success is True
        data = result.data
        assert data["name"] == "docs"
        assert data["points_count"] == 42
        assert data["topology"] == "dense"
        assert data["vectors"][""]["size"] == 384
        assert data["vectors"][""]["distance"] == "Cosine"
        assert data["quantization"] is None
        assert data["hnsw_config"]["m"] == 16
        assert data["hnsw_config"]["ef_construct"] == 100
        assert data["payload_schema"] is None
        assert data["sparse_vectors"] is None

    def test_show_collection_hybrid(self, executor, mock_client, mocker):
        from qdrant_client.models import (
            CollectionStatus,
            Distance,
            Modifier,
            SparseVectorParams,
            VectorParams,
        )

        mock_client.collection_exists.return_value = True

        mock_info = mocker.MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.points_count = 10
        mock_info.indexed_vectors_count = 10
        mock_info.segments_count = 1

        mock_info.config.params.vectors = {
            "dense": VectorParams(size=768, distance=Distance.COSINE),
        }
        mock_info.config.params.sparse_vectors = {
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        }
        mock_info.config.params.shard_number = 1
        mock_info.config.params.replication_factor = 1
        mock_info.config.params.write_consistency_factor = 1
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 100
        mock_info.config.hnsw_config.full_scan_threshold = None
        mock_info.config.hnsw_config.max_indexing_threads = None
        mock_info.config.hnsw_config.on_disk = None
        mock_info.config.hnsw_config.payload_m = None
        mock_info.config.quantization_config = None
        mock_info.payload_schema = {}

        mock_client.get_collection.return_value = mock_info

        node = ShowCollectionStmt(collection="hybrid_col")
        result = executor.execute(node)

        assert result.success is True
        data = result.data
        assert data["topology"] == "hybrid"
        assert data["vectors"]["dense"]["size"] == 768
        assert data["sparse_vectors"]["sparse"]["modifier"] == "idf"

    def test_show_collection_named_dense_is_not_reported_as_hybrid(self, executor, mock_client, mocker):
        from qdrant_client.models import (
            CollectionStatus,
            Distance,
            VectorParams,
        )

        mock_client.collection_exists.return_value = True

        mock_info = mocker.MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.points_count = 3
        mock_info.indexed_vectors_count = 3
        mock_info.segments_count = 1
        mock_info.config.params.vectors = {
            "body": VectorParams(size=384, distance=Distance.COSINE),
            "title": VectorParams(size=128, distance=Distance.DOT),
        }
        mock_info.config.params.sparse_vectors = None
        mock_info.config.params.shard_number = 1
        mock_info.config.params.replication_factor = 1
        mock_info.config.params.write_consistency_factor = 1
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 100
        mock_info.config.hnsw_config.full_scan_threshold = None
        mock_info.config.hnsw_config.max_indexing_threads = None
        mock_info.config.hnsw_config.on_disk = None
        mock_info.config.hnsw_config.payload_m = None
        mock_info.config.quantization_config = None
        mock_info.payload_schema = {}

        mock_client.get_collection.return_value = mock_info

        result = executor.execute(ShowCollectionStmt(collection="named_dense"))

        assert result.success is True
        assert result.data["topology"] == "dense"
        assert result.data["sparse_vectors"] is None

    def test_show_collection_with_payload_schema(self, executor, mock_client, mocker):
        from qdrant_client.models import (
            CollectionStatus,
            Distance,
            KeywordIndexParams,
            KeywordIndexType,
            PayloadSchemaType,
            VectorParams,
        )

        mock_client.collection_exists.return_value = True

        idx_info = mocker.MagicMock()
        idx_info.data_type = PayloadSchemaType.KEYWORD
        idx_info.params = KeywordIndexParams(
            type=KeywordIndexType.KEYWORD,
            is_tenant=True,
            on_disk=True,
        )

        mock_info = mocker.MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.points_count = 0
        mock_info.indexed_vectors_count = 0
        mock_info.segments_count = 0
        mock_info.config.params.vectors = VectorParams(size=384, distance=Distance.COSINE)
        mock_info.config.params.shard_number = 1
        mock_info.config.params.replication_factor = 1
        mock_info.config.params.write_consistency_factor = 1
        mock_info.config.params.sparse_vectors = None
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 100
        mock_info.config.hnsw_config.full_scan_threshold = None
        mock_info.config.hnsw_config.max_indexing_threads = None
        mock_info.config.hnsw_config.on_disk = None
        mock_info.config.hnsw_config.payload_m = None
        mock_info.config.quantization_config = None
        mock_info.payload_schema = {"category": idx_info}

        mock_client.get_collection.return_value = mock_info

        node = ShowCollectionStmt(collection="docs")
        result = executor.execute(node)

        assert result.success is True
        assert result.data["payload_schema"] == {
            "category": {
                "type": "keyword",
                "params": {"is_tenant": True, "on_disk": True},
            }
        }

    def test_show_collection_handles_missing_payload_schema(self, executor, mock_client, mocker):
        from qdrant_client.models import (
            CollectionStatus,
            Distance,
            VectorParams,
        )

        mock_client.collection_exists.return_value = True

        mock_info = mocker.MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.points_count = 0
        mock_info.indexed_vectors_count = 0
        mock_info.segments_count = 0
        mock_info.config.params.vectors = VectorParams(size=384, distance=Distance.COSINE)
        mock_info.config.params.shard_number = 1
        mock_info.config.params.replication_factor = 1
        mock_info.config.params.write_consistency_factor = 1
        mock_info.config.params.sparse_vectors = None
        mock_info.config.hnsw_config.m = 16
        mock_info.config.hnsw_config.ef_construct = 100
        mock_info.config.hnsw_config.full_scan_threshold = None
        mock_info.config.hnsw_config.max_indexing_threads = None
        mock_info.config.hnsw_config.on_disk = None
        mock_info.config.hnsw_config.payload_m = None
        mock_info.config.quantization_config = None
        mock_info.payload_schema = None

        mock_client.get_collection.return_value = mock_info

        result = executor.execute(ShowCollectionStmt(collection="docs"))

        assert result.success is True
        assert result.data["payload_schema"] is None

    def test_show_collection_nonexistent_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = ShowCollectionStmt(collection="ghost")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestScroll:
    def test_scroll_returns_points_and_next_offset(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        rec1 = mocker.MagicMock()
        rec1.id = "a"
        rec1.payload = {"text": "first"}
        rec2 = mocker.MagicMock()
        rec2.id = 2
        rec2.payload = {"text": "second"}
        mock_client.scroll.return_value = ([rec1, rec2], "next-1")

        node = ScrollStmt(collection="notes", limit=2)
        result = executor.execute(node)

        mock_client.scroll.assert_called_once_with(
            collection_name="notes",
            scroll_filter=None,
            limit=2,
            offset=None,
            with_payload=True,
            with_vectors=False,
        )
        assert result.success is True
        assert result.data == {
            "points": [
                {"id": "a", "payload": {"text": "first"}},
                {"id": "2", "payload": {"text": "second"}},
            ],
            "next_offset": "next-1",
        }

    def test_scroll_preserves_numeric_next_offset_type(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        rec = mocker.MagicMock()
        rec.id = 1
        rec.payload = {"text": "first"}
        mock_client.scroll.return_value = ([rec], 42)

        node = ScrollStmt(collection="notes", limit=1)
        result = executor.execute(node)

        assert result.success is True
        assert result.data["next_offset"] == 42

    def test_scroll_with_after_and_filter(self, executor, mock_client, mocker):
        from qql.ast_nodes import CompareExpr
        from qdrant_client.models import Filter

        mock_client.collection_exists.return_value = True
        mock_client.scroll.return_value = ([], None)

        node = ScrollStmt(
            collection="notes",
            limit=10,
            after="cursor-id",
            query_filter=CompareExpr(field="year", op=">=", value=2024),
        )
        executor.execute(node)

        kwargs = mock_client.scroll.call_args.kwargs
        assert kwargs["offset"] == "cursor-id"
        assert isinstance(kwargs["scroll_filter"], Filter)

    def test_scroll_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = ScrollStmt(collection="ghost", limit=5)
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestSelect:
    def test_select_by_id_returns_payload(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        rec = mocker.MagicMock()
        rec.id = "abc-123"
        rec.payload = {"text": "hello", "year": 2024}
        mock_client.retrieve.return_value = [rec]

        node = SelectStmt(collection="notes", point_id="abc-123")
        result = executor.execute(node)

        mock_client.retrieve.assert_called_once_with(
            collection_name="notes",
            ids=["abc-123"],
            with_payload=True,
            with_vectors=False,
        )
        assert result.success is True
        assert result.data == {"id": "abc-123", "payload": {"text": "hello", "year": 2024}}

    def test_select_not_found(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.return_value = []

        node = SelectStmt(collection="notes", point_id=7)
        result = executor.execute(node)

        assert result.success is True
        assert "not found" in result.message
        assert result.data is None

    def test_select_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = SelectStmt(collection="ghost", point_id="x")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestSearch:
    def test_search_calls_qdrant_query_points(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response
        node = SearchStmt(collection="notes", query_text="hello", limit=5, model=None)
        result = executor.execute(node)
        mock_client.query_points.assert_called_once()
        assert result.success is True

    def test_search_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = SearchStmt(collection="ghost", query_text="hi", limit=3, model=None)
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_search_with_exact_forwards_search_params(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            with_clause=SearchWith(exact=True),
        )
        executor.execute(node)

        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.exact is True

    def test_search_with_acorn_forwards_search_params(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            with_clause=SearchWith(hnsw_ef=128, acorn=True),
        )
        executor.execute(node)

        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.hnsw_ef == 128
        assert search_params.acorn.enable is True

    def test_search_with_indexed_only_and_quantization_forwards_search_params(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            with_clause=SearchWith(
                indexed_only=True,
                quantization=QuantizationSearchWith(
                    ignore=True,
                    rescore=False,
                    oversampling=2.5,
                ),
            ),
        )
        executor.execute(node)

        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.indexed_only is True
        assert search_params.quantization is not None
        assert search_params.quantization.ignore is True
        assert search_params.quantization.rescore is False
        assert search_params.quantization.oversampling == pytest.approx(2.5)

    def test_sparse_search_forwards_search_params(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            sparse_only=True,
            with_clause=SearchWith(exact=True, indexed_only=True),
        )
        executor.execute(node)

        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.exact is True
        assert search_params.indexed_only is True
    def test_dense_search_against_hybrid_collection_uses_dense_vector_name(
        self, executor, mock_client, mocker
    ):
        from qdrant_client.models import VectorParams

        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value.config.params.vectors = {
            "dense": VectorParams(size=384, distance="Cosine")
        }
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(collection="notes", query_text="hello", limit=5, model=None)
        executor.execute(node)

        assert mock_client.query_points.call_args.kwargs["using"] == "dense"

    def test_dense_search_with_mmr_uses_nearest_query(self, executor, mock_client, mocker):
        from qdrant_client.models import NearestQuery

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            with_clause=SearchWith(mmr_diversity=0.4, mmr_candidates=25),
        )
        executor.execute(node)

        query = mock_client.query_points.call_args.kwargs["query"]
        assert isinstance(query, NearestQuery)
        assert query.mmr is not None
        assert query.mmr.diversity == pytest.approx(0.4)
        assert query.mmr.candidates_limit == 25

    def test_hybrid_search_with_mmr_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            hybrid=True,
            with_clause=SearchWith(mmr_diversity=0.5),
        )
        with pytest.raises(QQLRuntimeError, match="MMR is not supported with USING HYBRID yet"):
            executor.execute(node)

    def test_sparse_search_with_mmr_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = SearchStmt(
            collection="notes",
            query_text="hello",
            limit=5,
            model=None,
            sparse_only=True,
            with_clause=SearchWith(mmr_diversity=0.5),
        )
        with pytest.raises(QQLRuntimeError, match="MMR is not supported with USING SPARSE yet"):
            executor.execute(node)


class TestRecommend:
    def test_recommend_calls_qdrant_query_points(self, executor, mock_client, mocker):
        from qdrant_client.models import RecommendQuery

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(collection="notes", positive_ids=("a",), limit=5)
        result = executor.execute(node)

        mock_client.query_points.assert_called_once()
        assert isinstance(mock_client.query_points.call_args.kwargs["query"], RecommendQuery)
        assert result.success is True
        assert "recommendation" in result.message

    def test_recommend_excludes_seed_ids_from_results_filter(
        self, executor, mock_client, mocker
    ):
        from qdrant_client.models import Filter, HasIdCondition

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a", 2),
            negative_ids=("x",),
            limit=5,
        )
        executor.execute(node)

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert isinstance(query_filter, Filter)
        assert isinstance(query_filter.must_not[0], HasIdCondition)
        assert query_filter.must_not[0].has_id == ["a", 2, "x"]

    def test_recommend_merges_where_filter_with_seed_exclusion(
        self, executor, mock_client, mocker
    ):
        from qdrant_client.models import Filter
        from qql.ast_nodes import CompareExpr

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            query_filter=CompareExpr(field="year", op=">", value=2020),
        )
        executor.execute(node)

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert isinstance(query_filter, Filter)
        assert query_filter.must is not None
        assert query_filter.must_not is not None

    def test_recommend_forwards_strategy(self, executor, mock_client, mocker):
        from qdrant_client.models import RecommendStrategy

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            strategy="best_score",
        )
        executor.execute(node)

        recommend = mock_client.query_points.call_args.kwargs["query"].recommend
        assert recommend.strategy == RecommendStrategy.BEST_SCORE

    def test_recommend_invalid_strategy_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            strategy="not-a-strategy",
        )
        with pytest.raises(QQLRuntimeError, match="Unknown recommend strategy"):
            executor.execute(node)

    def test_recommend_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = RecommendStmt(collection="ghost", positive_ids=("a",), limit=5)
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_recommend_forwards_offset(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes", positive_ids=("a",), limit=5, offset=10
        )
        executor.execute(node)
        assert mock_client.query_points.call_args.kwargs["offset"] == 10

    def test_recommend_forwards_score_threshold(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes", positive_ids=("a",), limit=5, score_threshold=0.5
        )
        executor.execute(node)
        assert mock_client.query_points.call_args.kwargs["score_threshold"] == pytest.approx(0.5)

    def test_recommend_forwards_using(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes", positive_ids=("a",), limit=5, using="sparse"
        )
        executor.execute(node)
        assert mock_client.query_points.call_args.kwargs["using"] == "sparse"

    def test_recommend_forwards_lookup_from(self, executor, mock_client, mocker):
        from qdrant_client.models import LookupLocation

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            lookup_from=("source", "dense"),
        )
        executor.execute(node)
        lookup = mock_client.query_points.call_args.kwargs["lookup_from"]
        assert isinstance(lookup, LookupLocation)
        assert lookup.collection == "source"
        assert lookup.vector == "dense"

    def test_recommend_forwards_lookup_from_without_vector(self, executor, mock_client, mocker):
        from qdrant_client.models import LookupLocation

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            lookup_from=("source", None),
        )
        executor.execute(node)
        lookup = mock_client.query_points.call_args.kwargs["lookup_from"]
        assert isinstance(lookup, LookupLocation)
        assert lookup.collection == "source"
        assert lookup.vector is None

    def test_recommend_forwards_search_params(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            with_clause=SearchWith(exact=True, hnsw_ef=128),
        )
        executor.execute(node)
        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.exact is True
        assert search_params.hnsw_ef == 128

    def test_recommend_forwards_indexed_only_and_quantization(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            with_clause=SearchWith(
                indexed_only=True,
                quantization=QuantizationSearchWith(rescore=True),
            ),
        )
        executor.execute(node)
        search_params = mock_client.query_points.call_args.kwargs["search_params"]
        assert search_params.indexed_only is True
        assert search_params.quantization is not None
        assert search_params.quantization.rescore is True

    def test_recommend_with_mmr_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = RecommendStmt(
            collection="notes",
            positive_ids=("a",),
            limit=5,
            with_clause=SearchWith(mmr_diversity=0.5),
        )
        with pytest.raises(QQLRuntimeError, match="MMR is supported only for SEARCH statements"):
            executor.execute(node)

    def test_recommend_offset_zero_passes_none(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        node = RecommendStmt(
            collection="notes", positive_ids=("a",), limit=5, offset=0
        )
        executor.execute(node)
        assert mock_client.query_points.call_args.kwargs["offset"] is None


class TestDelete:
    def test_delete_calls_qdrant_delete(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = DeleteStmt(collection="notes", point_id="abc-123")
        result = executor.execute(node)
        mock_client.delete.assert_called_once()
        assert result.success is True

    def test_delete_by_filter_calls_qdrant_delete_with_filter(self, executor, mock_client):
        from qdrant_client.models import Filter
        from qql.ast_nodes import CompareExpr

        mock_client.collection_exists.return_value = True
        node = DeleteStmt(
            collection="articles",
            query_filter=CompareExpr(field="category", op="=", value="archived"),
        )
        result = executor.execute(node)
        selector = mock_client.delete.call_args.kwargs["points_selector"]
        assert isinstance(selector, Filter)
        assert result.success is True

    def test_delete_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = DeleteStmt(collection="ghost", point_id="x")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)


class TestSearchWithFilter:
    """Tests for _build_qdrant_filter and filter pass-through in _execute_search."""

    def _search_node(self, query_filter=None):
        return SearchStmt(
            collection="docs", query_text="hello", limit=5, model=None,
            query_filter=query_filter,
        )

    def test_search_without_filter_passes_none_to_qdrant(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        executor.execute(self._search_node())

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is None

    def test_search_with_filter_passes_filter_to_qdrant(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        from qql.ast_nodes import CompareExpr
        node = self._search_node(query_filter=CompareExpr(field="cat", op="=", value="ai"))
        executor.execute(node)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is not None

    # ── _build_qdrant_filter unit tests (no Qdrant connection needed) ─────

    def test_build_equality(self, executor):
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from qql.ast_nodes import CompareExpr

        result = executor._wrap_as_filter(
            executor._build_qdrant_filter(CompareExpr(field="status", op="=", value="active"))
        )
        assert isinstance(result, Filter)
        fc = result.must[0]
        assert isinstance(fc, FieldCondition)
        assert fc.match == MatchValue(value="active")

    def test_build_not_equals(self, executor):
        from qdrant_client.models import Filter
        from qql.ast_nodes import CompareExpr

        result = executor._build_qdrant_filter(CompareExpr(field="s", op="!=", value="x"))
        assert isinstance(result, Filter)
        assert result.must_not is not None and len(result.must_not) == 1

    def test_build_range_gt(self, executor):
        from qdrant_client.models import FieldCondition
        from qql.ast_nodes import CompareExpr

        result = executor._build_qdrant_filter(CompareExpr(field="score", op=">", value=0.8))
        assert isinstance(result, FieldCondition)
        assert result.range.gt == pytest.approx(0.8)

    def test_build_range_gte(self, executor):
        from qdrant_client.models import FieldCondition
        from qql.ast_nodes import CompareExpr

        result = executor._build_qdrant_filter(CompareExpr(field="year", op=">=", value=2020))
        assert isinstance(result, FieldCondition)
        assert result.range.gte == 2020

    def test_build_range_lt(self, executor):
        from qdrant_client.models import FieldCondition
        from qql.ast_nodes import CompareExpr

        result = executor._build_qdrant_filter(CompareExpr(field="year", op="<", value=2024))
        assert isinstance(result, FieldCondition)
        assert result.range.lt == 2024

    def test_build_range_lte(self, executor):
        from qdrant_client.models import FieldCondition
        from qql.ast_nodes import CompareExpr

        result = executor._build_qdrant_filter(CompareExpr(field="year", op="<=", value=2023))
        assert isinstance(result, FieldCondition)
        assert result.range.lte == 2023

    def test_build_between(self, executor):
        from qdrant_client.models import FieldCondition
        from qql.ast_nodes import BetweenExpr

        result = executor._build_qdrant_filter(BetweenExpr(field="year", low=2018, high=2023))
        assert isinstance(result, FieldCondition)
        assert result.range.gte == 2018
        assert result.range.lte == 2023

    def test_build_in(self, executor):
        from qdrant_client.models import FieldCondition, MatchAny
        from qql.ast_nodes import InExpr

        result = executor._build_qdrant_filter(InExpr(field="status", values=("a", "b")))
        assert isinstance(result, FieldCondition)
        assert isinstance(result.match, MatchAny)

    def test_build_not_in(self, executor):
        from qdrant_client.models import FieldCondition, MatchExcept
        from qql.ast_nodes import NotInExpr

        result = executor._build_qdrant_filter(NotInExpr(field="status", values=("deleted",)))
        assert isinstance(result, FieldCondition)
        assert isinstance(result.match, MatchExcept)

    def test_build_is_null(self, executor):
        from qdrant_client.models import IsNullCondition
        from qql.ast_nodes import IsNullExpr

        result = executor._build_qdrant_filter(IsNullExpr(field="reviewer"))
        assert isinstance(result, IsNullCondition)

    def test_build_is_not_null(self, executor):
        from qdrant_client.models import Filter, IsNullCondition
        from qql.ast_nodes import IsNotNullExpr

        result = executor._build_qdrant_filter(IsNotNullExpr(field="reviewer"))
        assert isinstance(result, Filter)
        assert isinstance(result.must_not[0], IsNullCondition)

    def test_build_is_empty(self, executor):
        from qdrant_client.models import IsEmptyCondition
        from qql.ast_nodes import IsEmptyExpr

        result = executor._build_qdrant_filter(IsEmptyExpr(field="tags"))
        assert isinstance(result, IsEmptyCondition)

    def test_build_is_not_empty(self, executor):
        from qdrant_client.models import Filter, IsEmptyCondition
        from qql.ast_nodes import IsNotEmptyExpr

        result = executor._build_qdrant_filter(IsNotEmptyExpr(field="tags"))
        assert isinstance(result, Filter)
        assert isinstance(result.must_not[0], IsEmptyCondition)

    def test_build_match_text(self, executor):
        from qdrant_client.models import FieldCondition, MatchText
        from qql.ast_nodes import MatchTextExpr

        result = executor._build_qdrant_filter(MatchTextExpr(field="title", text="vector db"))
        assert isinstance(result, FieldCondition)
        assert isinstance(result.match, MatchText)
        assert result.match.text == "vector db"

    def test_build_match_any(self, executor):
        from qdrant_client.models import FieldCondition, MatchTextAny
        from qql.ast_nodes import MatchAnyExpr

        result = executor._build_qdrant_filter(MatchAnyExpr(field="title", text="nlp ai"))
        assert isinstance(result, FieldCondition)
        assert isinstance(result.match, MatchTextAny)

    def test_build_match_phrase(self, executor):
        from qdrant_client.models import FieldCondition, MatchPhrase
        from qql.ast_nodes import MatchPhraseExpr

        result = executor._build_qdrant_filter(MatchPhraseExpr(field="title", text="quick fox"))
        assert isinstance(result, FieldCondition)
        assert isinstance(result.match, MatchPhrase)

    def test_build_and(self, executor):
        from qdrant_client.models import Filter
        from qql.ast_nodes import AndExpr, CompareExpr

        expr = AndExpr(operands=(
            CompareExpr(field="a", op="=", value="x"),
            CompareExpr(field="b", op="=", value="y"),
        ))
        result = executor._build_qdrant_filter(expr)
        assert isinstance(result, Filter)
        assert len(result.must) == 2

    def test_build_or(self, executor):
        from qdrant_client.models import Filter
        from qql.ast_nodes import CompareExpr, OrExpr

        expr = OrExpr(operands=(
            CompareExpr(field="src", op="=", value="arxiv"),
            CompareExpr(field="src", op="=", value="ieee"),
        ))
        result = executor._build_qdrant_filter(expr)
        assert isinstance(result, Filter)
        assert len(result.should) == 2

    def test_build_not(self, executor):
        from qdrant_client.models import Filter
        from qql.ast_nodes import CompareExpr, NotExpr

        expr = NotExpr(operand=CompareExpr(field="st", op="=", value="draft"))
        result = executor._build_qdrant_filter(expr)
        assert isinstance(result, Filter)
        assert result.must_not is not None

    def test_wrap_as_filter_passthrough(self, executor):
        from qdrant_client.models import Filter

        f = Filter(must=[])
        assert executor._wrap_as_filter(f) is f

    def test_wrap_as_filter_wraps_field_condition(self, executor):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        fc = FieldCondition(key="x", match=MatchValue(value="y"))
        result = executor._wrap_as_filter(fc)
        assert isinstance(result, Filter)
        assert result.must[0] is fc


# ── Hybrid vector executor tests ──────────────────────────────────────────────

FAKE_SPARSE = {"indices": [1, 42, 100], "values": [0.22, 0.8, 0.3]}


@pytest.fixture
def mock_sparse_embedder(mocker):
    mock = mocker.MagicMock()
    mock.embed.return_value = FAKE_SPARSE
    mock.query_embed.return_value = FAKE_SPARSE
    mocker.patch("qql.executor.SparseEmbedder", return_value=mock)
    return mock


class TestHybridCreate:
    def test_create_hybrid_uses_named_vector_config(self, executor, mock_client):
        node = CreateCollectionStmt(collection="articles", hybrid=True)
        result = executor.execute(node)
        mock_client.create_collection.assert_called_once()
        kw = mock_client.create_collection.call_args.kwargs
        assert "sparse_vectors_config" in kw
        assert "dense" in kw["vectors_config"]
        assert "sparse" in kw["sparse_vectors_config"]
        assert result.success is True
        assert "hybrid" in result.message

    def test_create_hybrid_existing_collection_is_noop(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateCollectionStmt(collection="existing", hybrid=True)
        result = executor.execute(node)
        mock_client.create_collection.assert_not_called()
        assert "already exists" in result.message

    def test_create_non_hybrid_unchanged(self, executor, mock_client):
        from qdrant_client.models import VectorParams

        node = CreateCollectionStmt(collection="col", hybrid=False)
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["vectors_config"], VectorParams)
        assert "sparse_vectors_config" not in kw


class TestHybridInsert:
    def test_hybrid_insert_upsert_has_named_vectors(
        self, executor, mock_client, mock_sparse_embedder
    ):
        mock_client.collection_exists.return_value = True
        node = InsertStmt(
            collection="col", values={"text": "hello"}, model=None, hybrid=True
        )
        result = executor.execute(node)
        mock_client.upsert.assert_called_once()
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert result.success is True
        assert "hybrid" in result.message

    def test_hybrid_insert_sparse_is_SparseVector(
        self, executor, mock_client, mock_sparse_embedder
    ):
        from qdrant_client.models import SparseVector

        mock_client.collection_exists.return_value = True
        node = InsertStmt(
            collection="col", values={"text": "hello"}, model=None, hybrid=True
        )
        executor.execute(node)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert isinstance(point.vector["sparse"], SparseVector)

    def test_hybrid_insert_auto_creates_hybrid_collection(
        self, executor, mock_client, mock_sparse_embedder
    ):
        mock_client.collection_exists.return_value = False
        node = InsertStmt(
            collection="col", values={"text": "hello"}, model=None, hybrid=True
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert "sparse_vectors_config" in kw
        assert "dense" in kw["vectors_config"]

    def test_hybrid_insert_skips_create_when_exists(
        self, executor, mock_client, mock_sparse_embedder
    ):
        mock_client.collection_exists.return_value = True
        node = InsertStmt(
            collection="col", values={"text": "hello"}, model=None, hybrid=True
        )
        executor.execute(node)
        mock_client.create_collection.assert_not_called()

    def test_hybrid_insert_uses_custom_dense_model(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        mock_client.collection_exists.return_value = True
        node = InsertStmt(
            collection="col", values={"text": "hi"}, model="BAAI/bge-small-en-v1.5",
            hybrid=True,
        )
        executor.execute(node)
        # Embedder should have been called with the custom dense model name
        # Verify through the dense vector in the upsert call
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert "dense" in point.vector

    def test_hybrid_insert_uses_custom_sparse_model(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_sparse = mocker.MagicMock()
        mock_sparse.embed.return_value = FAKE_SPARSE
        sparse_cls = mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)
        node = InsertStmt(
            collection="col", values={"text": "hi"}, model=None,
            hybrid=True, sparse_model="prithivida/Splade_PP_en_v1",
        )
        executor.execute(node)
        sparse_cls.assert_called_once_with("prithivida/Splade_PP_en_v1")

    def test_non_hybrid_insert_uses_flat_vector(self, executor, mock_client):
        node = InsertStmt(
            collection="col", values={"text": "hello"}, model=None, hybrid=False
        )
        executor.execute(node)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert isinstance(point.vector, list)

    def test_hybrid_insert_missing_text_raises(
        self, executor, mock_client, mock_sparse_embedder
    ):
        node = InsertStmt(
            collection="col", values={"author": "alice"}, model=None, hybrid=True
        )
        with pytest.raises(QQLRuntimeError, match="'text' field"):
            executor.execute(node)


class TestHybridSearch:
    def test_hybrid_search_uses_prefetch(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="ml", limit=10, model=None, hybrid=True
        )
        result = executor.execute(node)
        mock_client.query_points.assert_called_once()
        kw = mock_client.query_points.call_args.kwargs
        assert "prefetch" in kw
        assert len(kw["prefetch"]) == 2
        assert result.success is True
        assert "hybrid" in result.message

    def test_hybrid_search_uses_rrf_fusion(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        from qdrant_client.models import Fusion, FusionQuery

        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, hybrid=True
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert isinstance(kw["query"], FusionQuery)
        assert kw["query"].fusion == Fusion.RRF

    def test_hybrid_search_uses_dbsf_fusion(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        from qdrant_client.models import Fusion, FusionQuery

        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col",
            query_text="q",
            limit=5,
            model=None,
            hybrid=True,
            fusion="dbsf",
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert isinstance(kw["query"], FusionQuery)
        assert kw["query"].fusion == Fusion.DBSF

    def test_hybrid_search_prefetch_limit_is_4x(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, hybrid=True
        )
        executor.execute(node)
        prefetches = mock_client.query_points.call_args.kwargs["prefetch"]
        assert all(p.limit == 20 for p in prefetches)

    def test_hybrid_search_prefetch_using_fields(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, hybrid=True
        )
        executor.execute(node)
        prefetches = mock_client.query_points.call_args.kwargs["prefetch"]
        usings = {p.using for p in prefetches}
        assert usings == {"dense", "sparse"}

    def test_hybrid_search_forwards_search_params_to_prefetch(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col",
            query_text="q",
            limit=5,
            model=None,
            hybrid=True,
            with_clause=SearchWith(exact=True, hnsw_ef=64),
        )
        executor.execute(node)

        prefetches = mock_client.query_points.call_args.kwargs["prefetch"]
        assert all(p.params is not None for p in prefetches)
        assert all(p.params.exact is True for p in prefetches)
        assert all(p.params.hnsw_ef == 64 for p in prefetches)

    def test_hybrid_search_with_where_filter(
        self, executor, mock_client, mock_sparse_embedder, mocker
    ):
        from qql.ast_nodes import CompareExpr
        from qdrant_client.models import Filter

        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, hybrid=True,
            query_filter=CompareExpr(field="year", op=">", value=2020),
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert kw.get("query_filter") is not None
        assert isinstance(kw["query_filter"], Filter)

    def test_hybrid_search_nonexistent_collection_raises(
        self, executor, mock_client, mock_sparse_embedder
    ):
        mock_client.collection_exists.return_value = False
        node = SearchStmt(
            collection="ghost", query_text="q", limit=5, model=None, hybrid=True
        )
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_non_hybrid_search_unchanged(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, hybrid=False
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert "prefetch" not in kw or kw.get("prefetch") is None

    def test_hybrid_search_uses_custom_sparse_model(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = FAKE_SPARSE
        sparse_cls = mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            hybrid=True, sparse_model="prithivida/Splade_PP_en_v1",
        )
        executor.execute(node)
        sparse_cls.assert_called_once_with("prithivida/Splade_PP_en_v1")


class TestEnsureCollectionHybridCompat:
    def test_named_vector_collection_skips_validation(self, executor, mock_client):
        from qdrant_client.models import VectorParams

        mock_client.collection_exists.return_value = True
        # Simulate a named-vector (hybrid) collection: vectors is a dict
        mock_client.get_collection.return_value.config.params.vectors = {
            "dense": VectorParams(size=384, distance="Cosine")
        }
        # Should not raise even with a different size argument
        executor._ensure_collection("hybrid_col", 384)
        mock_client.create_collection.assert_not_called()

    def test_unnamed_vector_mismatch_still_raises(self, executor, mock_client):
        from qdrant_client.models import VectorParams

        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value.config.params.vectors = VectorParams(
            size=768, distance="Cosine"
        )
        with pytest.raises(QQLRuntimeError, match="dimension mismatch"):
            executor._ensure_collection("col", 384)


FAKE_SPARSE = {"indices": [1, 42, 100], "values": [0.22, 0.8, 0.3]}


class TestRerankSearch:
    @pytest.fixture
    def mock_cross_encoder(self, mocker):
        mock = mocker.MagicMock()
        mock.rerank.return_value = [0.9, 0.3, 0.7]
        mocker.patch("qql.executor.CrossEncoderEmbedder", return_value=mock)
        return mock

    def _make_point(self, mocker, id_, score, text):
        p = mocker.MagicMock()
        p.id = id_
        p.score = score
        p.payload = {"text": text}
        return p

    def test_rerank_calls_cross_encoder_with_query_and_texts(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        pts = [
            self._make_point(mocker, "a", 0.9, "doc A"),
            self._make_point(mocker, "b", 0.5, "doc B"),
            self._make_point(mocker, "c", 0.7, "doc C"),
        ]
        mock_resp = mocker.MagicMock()
        mock_resp.points = pts
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="my query", limit=3, model=None, rerank=True
        )
        executor.execute(node)
        mock_cross_encoder.rerank.assert_called_once_with(
            "my query", ["doc A", "doc B", "doc C"]
        )

    def test_rerank_qdrant_fetches_multiplied_limit(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, rerank=True
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert kw["limit"] == 5 * 4  # _RERANK_FETCH_MULTIPLIER

    def test_rerank_results_sorted_by_cross_encoder_score(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        pts = [
            self._make_point(mocker, "a", 0.9, "doc A"),
            self._make_point(mocker, "b", 0.5, "doc B"),
            self._make_point(mocker, "c", 0.7, "doc C"),
        ]
        mock_resp = mocker.MagicMock()
        mock_resp.points = pts
        mock_client.query_points.return_value = mock_resp
        # scores: A→0.9, B→0.3, C→0.7  → sorted order: A, C, B
        mock_cross_encoder.rerank.return_value = [0.9, 0.3, 0.7]

        node = SearchStmt(
            collection="col", query_text="q", limit=3, model=None, rerank=True
        )
        result = executor.execute(node)
        ids = [r["id"] for r in result.data]
        assert ids == ["a", "c", "b"]

    def test_rerank_slices_to_limit(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        pts = [self._make_point(mocker, str(i), 0.5, f"doc {i}") for i in range(8)]
        mock_resp = mocker.MagicMock()
        mock_resp.points = pts
        mock_client.query_points.return_value = mock_resp
        mock_cross_encoder.rerank.return_value = [float(i) for i in range(8)]

        node = SearchStmt(
            collection="col", query_text="q", limit=3, model=None, rerank=True
        )
        result = executor.execute(node)
        assert len(result.data) == 3

    def test_rerank_message_contains_reranked(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp
        mock_cross_encoder.rerank.return_value = []

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, rerank=True
        )
        result = executor.execute(node)
        assert "reranked" in result.message

    def test_no_rerank_does_not_call_cross_encoder(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, rerank=False
        )
        executor.execute(node)
        mock_cross_encoder.rerank.assert_not_called()

    def test_no_rerank_uses_original_limit(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None, rerank=False
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert kw["limit"] == 5

    def test_rerank_custom_model_forwarded(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        mock_ce = mocker.MagicMock()
        mock_ce.rerank.return_value = []
        ce_cls = mocker.patch("qql.executor.CrossEncoderEmbedder", return_value=mock_ce)

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            rerank=True, rerank_model="my-custom/reranker",
        )
        executor.execute(node)
        ce_cls.assert_called_once_with("my-custom/reranker")

    def test_rerank_hybrid_search_message(
        self, executor, mock_client, mock_cross_encoder, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp
        mock_cross_encoder.rerank.return_value = []

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = FAKE_SPARSE
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            hybrid=True, rerank=True,
        )
        result = executor.execute(node)
        assert "hybrid" in result.message
        assert "reranked" in result.message


class TestSparseOnlySearch:
    @pytest.fixture
    def mock_sparse(self, mocker):
        mock = mocker.MagicMock()
        mock.query_embed.return_value = FAKE_SPARSE
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock)
        return mock

    def test_sparse_only_calls_query_embed(
        self, executor, mock_client, mock_sparse, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True,
        )
        executor.execute(node)
        mock_sparse.query_embed.assert_called_once_with("q")

    def test_sparse_only_queries_sparse_vector_name(
        self, executor, mock_client, mock_sparse, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True,
        )
        executor.execute(node)
        kw = mock_client.query_points.call_args.kwargs
        assert kw["using"] == "sparse"

    def test_sparse_only_message_contains_sparse(
        self, executor, mock_client, mock_sparse, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True,
        )
        result = executor.execute(node)
        assert "sparse" in result.message

    def test_sparse_only_uses_custom_model(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = FAKE_SPARSE
        sparse_cls = mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True, sparse_model="prithivida/Splade_PP_en_v1",
        )
        executor.execute(node)
        sparse_cls.assert_called_once_with("prithivida/Splade_PP_en_v1")

    def test_sparse_only_uses_default_model_when_none(
        self, executor, mock_client, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = FAKE_SPARSE
        sparse_cls = mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)
        # Make DEFAULT_MODEL on the mock class resolve to the real value so the
        # executor's `node.sparse_model or SparseEmbedder.DEFAULT_MODEL` uses it.
        sparse_cls.DEFAULT_MODEL = "Qdrant/bm25"

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True,
        )
        executor.execute(node)
        sparse_cls.assert_called_once_with("Qdrant/bm25")

    def test_sparse_only_with_rerank_message(
        self, executor, mock_client, mock_sparse, mocker
    ):
        mock_client.collection_exists.return_value = True
        mock_resp = mocker.MagicMock()
        mock_resp.points = []
        mock_client.query_points.return_value = mock_resp

        mock_ce = mocker.MagicMock()
        mock_ce.rerank.return_value = []
        mocker.patch("qql.executor.CrossEncoderEmbedder", return_value=mock_ce)

        node = SearchStmt(
            collection="col", query_text="q", limit=5, model=None,
            sparse_only=True, rerank=True,
        )
        result = executor.execute(node)
        assert "sparse" in result.message
        assert "reranked" in result.message


# ── TestQuantizeCreate ────────────────────────────────────────────────────────


class TestQuantizeCreate:
    # ── Scalar ────────────────────────────────────────────────────────────

    def test_scalar_passes_scalar_quantization(self, executor, mock_client):
        from qdrant_client.models import ScalarQuantization
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), ScalarQuantization)

    def test_scalar_type_is_int8(self, executor, mock_client):
        from qdrant_client.models import ScalarType
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].scalar.type == ScalarType.INT8

    def test_scalar_quantile_none_by_default(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].scalar.quantile is None

    def test_scalar_explicit_quantile(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR, quantile=0.95),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].scalar.quantile == pytest.approx(0.95)

    def test_scalar_always_ram_true(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR, always_ram=True),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].scalar.always_ram is True

    # ── Binary ────────────────────────────────────────────────────────────

    def test_binary_passes_binary_quantization(self, executor, mock_client):
        from qdrant_client.models import BinaryQuantization
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.BINARY),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), BinaryQuantization)

    def test_binary_always_ram(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.BINARY, always_ram=True),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].binary.always_ram is True

    # ── Product ───────────────────────────────────────────────────────────

    def test_product_passes_product_quantization(self, executor, mock_client):
        from qdrant_client.models import ProductQuantization
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.PRODUCT),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), ProductQuantization)

    def test_product_uses_x4_compression(self, executor, mock_client):
        from qdrant_client.models import CompressionRatio
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.PRODUCT),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].product.compression == CompressionRatio.X4

    # ── Combined with hybrid ──────────────────────────────────────────────

    def test_hybrid_with_quantization_has_both_configs(self, executor, mock_client):
        from qdrant_client.models import ScalarQuantization
        node = CreateCollectionStmt(
            collection="articles",
            hybrid=True,
            quantization=QuantizationConfig(type=QuantizationType.SCALAR),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), ScalarQuantization)
        assert "sparse_vectors_config" in kw

    def test_hybrid_with_quantization_vectors_config_is_named_dict(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            hybrid=True,
            quantization=QuantizationConfig(type=QuantizationType.BINARY),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw["vectors_config"], dict)
        assert "dense" in kw["vectors_config"]

    # ── No quantization — backward compatibility ──────────────────────────

    def test_no_quantization_omits_kwarg(self, executor, mock_client):
        node = CreateCollectionStmt(collection="articles")
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert "quantization_config" not in kw

    # ── Result message ────────────────────────────────────────────────────

    def test_result_message_includes_quantization_type(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.SCALAR),
        )
        result = executor.execute(node)
        assert "scalar" in result.message

    def test_result_message_no_quantization_suffix_when_absent(self, executor, mock_client):
        node = CreateCollectionStmt(collection="articles")
        result = executor.execute(node)
        assert "quantization" not in result.message


class TestTurboQuantCreate:
    """Executor tests for QUANTIZE TURBO — verifies correct SDK objects are built."""

    @pytest.fixture
    def executor(self, cfg, mock_client):
        return Executor(mock_client, cfg)

    # ── TurboQuantization object is produced ──────────────────────────────

    def test_turbo_passes_turbo_quantization(self, executor, mock_client):
        from qdrant_client.models import TurboQuantization
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), TurboQuantization)

    def test_turbo_default_bits_is_none(self, executor, mock_client):
        """When BITS is omitted, bits must be None — preserving omission so the
        SDK/server applies its own default rather than QQL forcing BITS4."""
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.bits is None

    def test_turbo_bits2(self, executor, mock_client):
        from qdrant_client.models import TurboQuantBitSize
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO, turbo_bits=2.0),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.bits == TurboQuantBitSize.BITS2

    def test_turbo_bits1_5(self, executor, mock_client):
        from qdrant_client.models import TurboQuantBitSize
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO, turbo_bits=1.5),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.bits == TurboQuantBitSize.BITS1_5

    def test_turbo_bits1(self, executor, mock_client):
        from qdrant_client.models import TurboQuantBitSize
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO, turbo_bits=1.0),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.bits == TurboQuantBitSize.BITS1

    def test_turbo_always_ram_true(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO, always_ram=True),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.always_ram is True

    def test_turbo_always_ram_false_by_default(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert kw["quantization_config"].turbo.always_ram is False

    def test_turbo_hybrid_collection_has_both_configs(self, executor, mock_client):
        from qdrant_client.models import TurboQuantization
        node = CreateCollectionStmt(
            collection="articles",
            hybrid=True,
            quantization=QuantizationConfig(type=QuantizationType.TURBO),
        )
        executor.execute(node)
        kw = mock_client.create_collection.call_args.kwargs
        assert isinstance(kw.get("quantization_config"), TurboQuantization)
        assert "sparse_vectors_config" in kw

    def test_turbo_result_message_includes_turbo(self, executor, mock_client):
        node = CreateCollectionStmt(
            collection="articles",
            quantization=QuantizationConfig(type=QuantizationType.TURBO),
        )
        result = executor.execute(node)
        assert "turbo" in result.message

    def test_turbo_invalid_bits_at_executor_raises(self, executor, mock_client):
        """An unexpected turbo_bits value that bypasses parser validation must
        raise QQLRuntimeError explicitly instead of silently coercing to BITS4."""
        from qql.exceptions import QQLRuntimeError as QQLErr
        qc = QuantizationConfig(type=QuantizationType.TURBO, turbo_bits=3.0)
        with pytest.raises(QQLErr, match="Unsupported TURBO bit depth"):
            executor._build_quantization_config(qc)


# ── New feature executor tests ────────────────────────────────────────────────

class TestSearchGroupBy:
    def test_group_by_calls_query_points_groups(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_group = mocker.MagicMock()
        mock_group.id = "tech"
        mock_hit = mocker.MagicMock()
        mock_hit.id = "abc-123"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "hello"}
        mock_group.hits = [mock_hit]
        mock_response = mocker.MagicMock()
        mock_response.groups = [mock_group]
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles", query_text="ai", limit=5, model=None,
            group_by="category", group_size=3,
        )
        result = executor.execute(node)
        mock_client.query_points_groups.assert_called_once()
        assert result.success is True

    def test_group_by_result_structure(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_group = mocker.MagicMock()
        mock_group.id = "science"
        mock_hit = mocker.MagicMock()
        mock_hit.id = "xyz-456"
        mock_hit.score = 0.88
        mock_hit.payload = {"text": "deep learning"}
        mock_group.hits = [mock_hit]
        mock_response = mocker.MagicMock()
        mock_response.groups = [mock_group]
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles", query_text="ml", limit=3, model=None,
            group_by="field", group_size=2,
        )
        result = executor.execute(node)
        assert len(result.data) == 1
        assert result.data[0]["group_id"] == "science"
        assert len(result.data[0]["hits"]) == 1
        assert result.data[0]["hits"][0]["score"] == 0.88

    def test_group_by_message_contains_field_name(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            group_by="category",
        )
        result = executor.execute(node)
        assert "category" in result.message

    def test_group_by_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = SearchStmt(
            collection="ghost", query_text="q", limit=5, model=None,
            group_by="field",
        )
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_group_by_passes_group_size_to_qdrant(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            group_by="tag", group_size=7,
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        assert kwargs["group_size"] == 7
        assert kwargs["group_by"] == "tag"

    def test_group_by_hybrid_uses_query_points_groups(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        mock_sparse_embedder = mocker.MagicMock()
        mock_sparse_embedder.query_embed.return_value = {"indices": [0, 1], "values": [0.5, 0.5]}
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse_embedder)

        node = SearchStmt(
            collection="articles", query_text="q", limit=3, model=None,
            hybrid=True, group_by="category", group_size=2,
        )
        executor.execute(node)
        mock_client.query_points_groups.assert_called_once()
        kwargs = mock_client.query_points_groups.call_args.kwargs
        assert kwargs["group_by"] == "category"
        assert "prefetch" in kwargs

    def test_group_by_dense_with_mmr_uses_nearest_query(self, executor, mock_client, mocker):
        from qdrant_client.models import NearestQuery

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles",
            query_text="ai",
            limit=5,
            model=None,
            group_by="category",
            with_clause=SearchWith(mmr_diversity=0.35, mmr_candidates=40),
        )
        executor.execute(node)
        query = mock_client.query_points_groups.call_args.kwargs["query"]
        assert isinstance(query, NearestQuery)
        assert query.mmr is not None
        assert query.mmr.diversity == pytest.approx(0.35)
        assert query.mmr.candidates_limit == 40


class TestUpdateVector:
    def test_update_vector_calls_update_vectors(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        node = UpdateVectorStmt(
            collection="articles", point_id="abc-123", vector=(0.1, 0.2, 0.3)
        )
        result = executor.execute(node)
        mock_client.update_vectors.assert_called_once()
        assert result.success is True

    def test_update_vector_passes_correct_point_id(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value.config.params.vectors = {}  # non-dict → unnamed
        node = UpdateVectorStmt(
            collection="notes", point_id=42, vector=(0.5, 0.6)
        )
        executor.execute(node)
        call_kwargs = mock_client.update_vectors.call_args.kwargs
        points = call_kwargs["points"]
        assert len(points) == 1
        assert points[0].id == 42

    def test_update_vector_nonexistent_collection_raises(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = False
        node = UpdateVectorStmt(collection="ghost", point_id=1, vector=(0.1,))
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_update_vector_result_success_message(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        node = UpdateVectorStmt(collection="articles", point_id="id-1", vector=(0.1, 0.2))
        result = executor.execute(node)
        assert result.success is True
        assert "id-1" in result.message

    def test_update_vector_passes_wait_true(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        node = UpdateVectorStmt(collection="articles", point_id=1, vector=(0.1,))
        executor.execute(node)
        kwargs = mock_client.update_vectors.call_args.kwargs
        assert kwargs.get("wait") is True


class TestUpdatePayload:
    def test_update_payload_by_id_calls_set_payload(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles", point_id="abc-123", payload={"year": 2025}
        )
        result = executor.execute(node)
        mock_client.set_payload.assert_called_once()
        assert result.success is True

    def test_update_payload_by_filter_calls_set_payload_with_filter(
        self, executor, mock_client
    ):
        from qql.ast_nodes import UpdatePayloadStmt, CompareExpr
        from qdrant_client.models import Filter
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles",
            payload={"status": "published"},
            query_filter=CompareExpr(field="category", op="=", value="draft"),
        )
        result = executor.execute(node)
        mock_client.set_payload.assert_called_once()
        kwargs = mock_client.set_payload.call_args.kwargs
        assert isinstance(kwargs["points"], Filter)
        assert result.success is True

    def test_update_payload_nonexistent_collection_raises(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = False
        node = UpdatePayloadStmt(collection="ghost", point_id=1, payload={"x": 1})
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)

    def test_update_payload_result_success_message(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles", point_id="id-99", payload={"tag": "ai"}
        )
        result = executor.execute(node)
        assert result.success is True
        assert "id-99" in result.message

    def test_update_payload_passes_correct_payload(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = True
        payload = {"title": "New", "score": 0.9}
        node = UpdatePayloadStmt(collection="articles", point_id=1, payload=payload)
        executor.execute(node)
        kwargs = mock_client.set_payload.call_args.kwargs
        assert kwargs["payload"] == payload

    def test_update_payload_passes_wait_true(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(collection="articles", point_id=1, payload={"x": 1})
        executor.execute(node)
        kwargs = mock_client.set_payload.call_args.kwargs
        assert kwargs.get("wait") is True


# ── PR #28 review gap fixes ───────────────────────────────────────────────────

class TestSearchGroupBySparse:
    """Gap 1 & 6 — sparse-only grouped search must use the sparse path."""

    def test_sparse_only_grouped_calls_query_points_groups(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = {"indices": [0, 1], "values": [0.5, 0.5]}
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            sparse_only=True, group_by="category", group_size=3,
        )
        executor.execute(node)
        mock_client.query_points_groups.assert_called_once()
        kwargs = mock_client.query_points_groups.call_args.kwargs
        assert kwargs.get("using") == "sparse"
        # Must NOT have called dense Embedder
        from qql.embedder import Embedder as _Embedder  # noqa: F401
        # mock_embedder fixture patches Embedder; query_points not called confirms no dense path
        mock_client.query_points.assert_not_called()

    def test_sparse_only_grouped_label_in_message(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = {"indices": [0], "values": [1.0]}
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            sparse_only=True, group_by="tag", group_size=2,
        )
        result = executor.execute(node)
        assert "sparse" in result.message
        assert "grouped" in result.message


class TestSearchGroupByAdvanced:
    """Gaps 7 & 8 — fusion and search params forwarding in grouped search."""

    def test_grouped_hybrid_fusion_dbsf(self, executor, mock_client, mocker):
        from qdrant_client.models import Fusion
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        mock_sparse = mocker.MagicMock()
        mock_sparse.query_embed.return_value = {"indices": [0], "values": [1.0]}
        mocker.patch("qql.executor.SparseEmbedder", return_value=mock_sparse)

        node = SearchStmt(
            collection="articles", query_text="q", limit=3, model=None,
            hybrid=True, fusion="dbsf", group_by="category", group_size=2,
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        fusion_query = kwargs.get("query")
        assert fusion_query is not None
        assert fusion_query.fusion == Fusion.DBSF

    def test_grouped_search_params_with_clause_forwarded(self, executor, mock_client, mocker):
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            with_clause=SearchWith(exact=True), group_by="category",
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        assert kwargs.get("search_params") is not None


class TestUpdateVectorVectorShape:
    """Gaps 12 & 13 — verify exact vector shape sent to Qdrant for named/unnamed collections."""

    def test_update_vector_unnamed_collection_sends_plain_list(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        # Unnamed collection: get_collection returns non-dict vectors
        info = mock_client.get_collection.return_value
        info.config.params.vectors = [None]  # list → not a dict → unnamed

        node = UpdateVectorStmt(collection="articles", point_id=1, vector=(0.1, 0.2, 0.3))
        executor.execute(node)
        kwargs = mock_client.update_vectors.call_args.kwargs
        pv = kwargs["points"][0]
        assert isinstance(pv.vector, list)
        assert pv.vector == [0.1, 0.2, 0.3]

    def test_update_vector_named_collection_sends_dict(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        # Named collection: get_collection returns dict vectors
        info = mock_client.get_collection.return_value
        info.config.params.vectors = {"dense": object(), "sparse": object()}  # dict → named

        node = UpdateVectorStmt(collection="articles", point_id="id-1", vector=(0.5, 0.6))
        executor.execute(node)
        kwargs = mock_client.update_vectors.call_args.kwargs
        pv = kwargs["points"][0]
        assert isinstance(pv.vector, dict)
        assert "dense" in pv.vector
        assert pv.vector["dense"] == [0.5, 0.6]

    def test_update_vector_exact_values_preserved(self, executor, mock_client):
        from qql.ast_nodes import UpdateVectorStmt
        mock_client.collection_exists.return_value = True
        info = mock_client.get_collection.return_value
        info.config.params.vectors = [None]  # unnamed

        vec = (0.11, 0.22, 0.33, 0.44)
        node = UpdateVectorStmt(collection="articles", point_id=99, vector=vec)
        executor.execute(node)
        kwargs = mock_client.update_vectors.call_args.kwargs
        assert kwargs["points"][0].vector == list(vec)


class TestUpdatePayloadMessages:
    """Gaps 17 — assert specific message text for both update-payload branches."""

    def test_filter_based_update_message_contains_filter_based(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt, CompareExpr
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles",
            payload={"status": "done"},
            query_filter=CompareExpr(field="year", op="<", value=2020),
        )
        result = executor.execute(node)
        assert "filter-based" in result.message

    def test_id_based_update_message_contains_point_id(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles", point_id="abc-999", payload={"tag": "ai"}
        )
        result = executor.execute(node)
        assert "abc-999" in result.message

    def test_filter_based_set_payload_receives_filter_object(self, executor, mock_client):
        from qql.ast_nodes import UpdatePayloadStmt, CompareExpr
        from qdrant_client.models import Filter
        mock_client.collection_exists.return_value = True
        node = UpdatePayloadStmt(
            collection="articles",
            payload={"x": 1},
            query_filter=CompareExpr(field="cat", op="=", value="tech"),
        )
        executor.execute(node)
        kwargs = mock_client.set_payload.call_args.kwargs
        # SDK verified: PointsSelector accepts rest.Filter — must receive Filter, not a list
        assert isinstance(kwargs["points"], Filter)


# ── Round-2 review gap fixes ──────────────────────────────────────────────────

class TestGroupedCustomModelForwarding:
    """Gap 9 (round 2) — grouped search must forward custom model names to the embedders."""

    def test_grouped_dense_custom_model_forwarded(self, executor, mock_client, mocker):
        """Dense grouped search with USING MODEL must instantiate Embedder with that model."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response
        mock_client.get_collection.return_value.config.params.vectors = None  # unnamed

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_instance = mocker.MagicMock()
        embedder_instance.embed.return_value = [0.1] * 384
        embedder_cls.return_value = embedder_instance

        node = SearchStmt(
            collection="articles", query_text="q", limit=5,
            model="BAAI/bge-base-en-v1.5",
            group_by="category", group_size=3,
        )
        executor.execute(node)
        # Embedder must have been instantiated with the custom model name
        embedder_cls.assert_called_once_with("BAAI/bge-base-en-v1.5")

    def test_grouped_hybrid_custom_dense_model_forwarded(self, executor, mock_client, mocker):
        """Hybrid grouped search must instantiate Embedder with the custom dense model."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.1] * 384

        sparse_cls = mocker.patch("qql.executor.SparseEmbedder")
        sparse_cls.return_value.query_embed.return_value = {"indices": [0], "values": [1.0]}

        node = SearchStmt(
            collection="articles", query_text="q", limit=5,
            model="BAAI/bge-large-en-v1.5", hybrid=True,
            group_by="category", group_size=2,
        )
        executor.execute(node)
        embedder_cls.assert_called_once_with("BAAI/bge-large-en-v1.5")

    def test_grouped_hybrid_custom_sparse_model_forwarded(self, executor, mock_client, mocker):
        """Hybrid grouped search must instantiate SparseEmbedder with the custom sparse model."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.1] * 384

        sparse_cls = mocker.patch("qql.executor.SparseEmbedder")
        sparse_cls.return_value.query_embed.return_value = {"indices": [0], "values": [1.0]}

        node = SearchStmt(
            collection="articles", query_text="q", limit=5,
            model=None, hybrid=True, sparse_model="prithivida/Splade_PP_en_v1",
            group_by="category", group_size=2,
        )
        executor.execute(node)
        sparse_cls.assert_called_once_with("prithivida/Splade_PP_en_v1")


class TestGroupedSearchParamsDepth:
    """Gap 8 (round 2) — verify hnsw_ef/acorn values are actually forwarded in grouped search."""

    def test_grouped_search_hnsw_ef_value_forwarded(self, executor, mock_client, mocker):
        """search_params for dense grouped search must carry the hnsw_ef value."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response
        mock_client.get_collection.return_value.config.params.vectors = None  # unnamed

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.1] * 384

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            with_clause=SearchWith(hnsw_ef=256),
            group_by="category", group_size=3,
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        sp = kwargs.get("search_params")
        assert sp is not None
        assert sp.hnsw_ef == 256

    def test_grouped_search_exact_value_forwarded(self, executor, mock_client, mocker):
        """search_params for dense grouped search must carry exact=True when set."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response
        mock_client.get_collection.return_value.config.params.vectors = None  # unnamed

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.1] * 384

        node = SearchStmt(
            collection="articles", query_text="q", limit=5, model=None,
            with_clause=SearchWith(exact=True),
            group_by="category", group_size=3,
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        sp = kwargs.get("search_params")
        assert sp is not None
        assert sp.exact is True

    def test_grouped_hybrid_prefetch_params_forwarded(self, executor, mock_client, mocker):
        """Hybrid grouped search must forward search_params into each Prefetch.params."""
        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.1] * 384

        sparse_cls = mocker.patch("qql.executor.SparseEmbedder")
        sparse_cls.return_value.query_embed.return_value = {"indices": [0], "values": [1.0]}

        node = SearchStmt(
            collection="articles", query_text="q", limit=5,
            model=None, hybrid=True, with_clause=SearchWith(hnsw_ef=128),
            group_by="category", group_size=2,
        )
        executor.execute(node)
        kwargs = mock_client.query_points_groups.call_args.kwargs
        prefetch_list = kwargs.get("prefetch")
        assert prefetch_list is not None and len(prefetch_list) == 2
        for pf in prefetch_list:
            assert pf.params is not None
            assert pf.params.hnsw_ef == 128


class TestBuildHybridVectorsHelper:
    """Gap 10 (round 2) — _build_hybrid_vectors() is the single source of truth for both paths."""

    def test_helper_returns_list_and_sparse_vector(self, executor, mocker):
        """_build_hybrid_vectors must return (list[float], SparseVector)."""
        from qdrant_client.models import SparseVector

        embedder_cls = mocker.patch("qql.executor.Embedder")
        embedder_cls.return_value.embed.return_value = [0.5] * 4

        sparse_cls = mocker.patch("qql.executor.SparseEmbedder")
        sparse_cls.return_value.query_embed.return_value = {"indices": [1, 3], "values": [0.8, 0.2]}

        dense, sparse = executor._build_hybrid_vectors("test query", "dense-model", "sparse-model")
        assert dense == [0.5] * 4
        assert isinstance(sparse, SparseVector)
        assert sparse.indices == [1, 3]
        assert sparse.values == [0.8, 0.2]
        embedder_cls.assert_called_once_with("dense-model")
        sparse_cls.assert_called_once_with("sparse-model")

    def test_flat_hybrid_search_uses_build_hybrid_vectors(self, executor, mock_client, mocker):
        """The flat hybrid path must call _build_hybrid_vectors (not inline Embedder calls)."""
        from qdrant_client.models import SparseVector

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        # Patch the shared helper on the executor instance
        helper = mocker.patch.object(
            executor, "_build_hybrid_vectors",
            return_value=([0.1] * 384, SparseVector(indices=[0], values=[1.0])),
        )

        node = SearchStmt(
            collection="articles", query_text="hello", limit=5,
            model=None, hybrid=True,
        )
        executor.execute(node)
        helper.assert_called_once()

    def test_grouped_hybrid_search_uses_build_hybrid_vectors(self, executor, mock_client, mocker):
        """The grouped hybrid path must call _build_hybrid_vectors (not inline Embedder calls)."""
        from qdrant_client.models import SparseVector

        mock_client.collection_exists.return_value = True
        mock_response = mocker.MagicMock()
        mock_response.groups = []
        mock_client.query_points_groups.return_value = mock_response

        helper = mocker.patch.object(
            executor, "_build_hybrid_vectors",
            return_value=([0.1] * 384, SparseVector(indices=[0], values=[1.0])),
        )

        node = SearchStmt(
            collection="articles", query_text="hello", limit=5,
            model=None, hybrid=True, group_by="category",
        )
        executor.execute(node)
        helper.assert_called_once()
