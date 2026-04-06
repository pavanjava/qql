import pytest

from qql.ast_nodes import (
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertStmt,
    SearchStmt,
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


class TestCreate:
    def test_create_new_collection(self, executor, mock_client):
        node = CreateCollectionStmt(collection="new_col")
        result = executor.execute(node)
        mock_client.create_collection.assert_called_once()
        assert result.success is True

    def test_create_existing_collection_is_noop(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = CreateCollectionStmt(collection="existing")
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


class TestDelete:
    def test_delete_calls_qdrant_delete(self, executor, mock_client):
        mock_client.collection_exists.return_value = True
        node = DeleteStmt(collection="notes", point_id="abc-123")
        result = executor.execute(node)
        mock_client.delete.assert_called_once()
        assert result.success is True

    def test_delete_nonexistent_collection_raises(self, executor, mock_client):
        mock_client.collection_exists.return_value = False
        node = DeleteStmt(collection="ghost", point_id="x")
        with pytest.raises(QQLRuntimeError, match="does not exist"):
            executor.execute(node)
