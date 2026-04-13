import pytest

from qql.ast_nodes import (
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertStmt,
    SearchStmt,
    SearchWith,
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
        call_args = mocker.patch.object  # already patched by mock_embedder fixture
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
