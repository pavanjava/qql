import pytest

from qql.ast_nodes import (
    AndExpr,
    BetweenExpr,
    CompareExpr,
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
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
    SearchWith,
    ShowCollectionsStmt,
)
from qql.exceptions import QQLSyntaxError
from qql.lexer import Lexer
from qql.parser import Parser


def parse(query: str):
    tokens = Lexer().tokenize(query)
    return Parser(tokens).parse()


class TestInsert:
    def test_basic_insert(self):
        node = parse("INSERT INTO COLLECTION notes VALUES {'text': 'hello'}")
        assert isinstance(node, InsertStmt)
        assert node.collection == "notes"
        assert node.values == {"text": "hello"}
        assert node.model is None

    def test_insert_with_metadata(self):
        node = parse("INSERT INTO COLLECTION notes VALUES {'text': 'hi', 'author': 'alice'}")
        assert node.values["author"] == "alice"
        assert node.values["text"] == "hi"

    def test_insert_with_model(self):
        node = parse(
            "INSERT INTO COLLECTION notes VALUES {'text': 'hi'} "
            "USING MODEL 'sentence-transformers/all-MiniLM-L6-v2'"
        )
        assert node.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_insert_case_insensitive(self):
        node = parse("insert into collection notes values {'text': 'hello'}")
        assert isinstance(node, InsertStmt)

    def test_insert_nested_dict(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'meta': {'src': 'web'}}")
        assert node.values["meta"] == {"src": "web"}

    def test_insert_list_value(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'tags': ['a', 'b']}")
        assert node.values["tags"] == ["a", "b"]

    def test_insert_integer_value(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'count': 42}")
        assert node.values["count"] == 42

    def test_insert_float_value(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'score': 0.9}")
        assert node.values["score"] == pytest.approx(0.9)

    def test_insert_bool_value(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'active': true}")
        assert node.values["active"] is True

    def test_insert_null_value(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'x', 'ref': null}")
        assert node.values["ref"] is None

    def test_missing_text_is_not_parser_error(self):
        # Schema validation is the executor's job, not the parser's
        node = parse("INSERT INTO COLLECTION col VALUES {'author': 'bob'}")
        assert isinstance(node, InsertStmt)
        assert "text" not in node.values


class TestCreate:
    def test_create_collection(self):
        node = parse("CREATE COLLECTION my_col")
        assert isinstance(node, CreateCollectionStmt)
        assert node.collection == "my_col"


class TestDrop:
    def test_drop_collection(self):
        node = parse("DROP COLLECTION my_col")
        assert isinstance(node, DropCollectionStmt)
        assert node.collection == "my_col"


class TestShow:
    def test_show_collections(self):
        node = parse("SHOW COLLECTIONS")
        assert isinstance(node, ShowCollectionsStmt)


class TestSearch:
    def test_basic_search(self):
        node = parse("SEARCH notes SIMILAR TO 'hello world' LIMIT 5")
        assert isinstance(node, SearchStmt)
        assert node.collection == "notes"
        assert node.query_text == "hello world"
        assert node.limit == 5
        assert node.model is None

    def test_search_with_model(self):
        node = parse("SEARCH notes SIMILAR TO 'hi' LIMIT 3 USING MODEL 'my-model'")
        assert node.model == "my-model"


class TestDelete:
    def test_delete_by_string_id(self):
        node = parse("DELETE FROM notes WHERE id = 'abc-123'")
        assert isinstance(node, DeleteStmt)
        assert node.collection == "notes"
        assert node.point_id == "abc-123"

    def test_delete_by_integer_id(self):
        node = parse("DELETE FROM notes WHERE id = 99")
        assert isinstance(node, DeleteStmt)
        assert node.point_id == 99


class TestErrors:
    def test_unknown_keyword(self):
        with pytest.raises(QQLSyntaxError):
            parse("SELECT * FROM foo")

    def test_missing_collection_name(self):
        with pytest.raises(QQLSyntaxError):
            parse("INSERT INTO COLLECTION VALUES {'text': 'x'}")

    def test_empty_input(self):
        with pytest.raises(QQLSyntaxError):
            parse("")


class TestSearchWithWhere:
    def test_no_where_clause(self):
        node = parse("SEARCH docs SIMILAR TO 'ml' LIMIT 5")
        assert node.query_filter is None

    def test_equality_filter(self):
        node = parse("SEARCH docs SIMILAR TO 'ml' LIMIT 5 WHERE category = 'paper'")
        f = node.query_filter
        assert isinstance(f, CompareExpr)
        assert f.field == "category"
        assert f.op == "="
        assert f.value == "paper"

    def test_not_equals_filter(self):
        node = parse("SEARCH docs SIMILAR TO 'ml' LIMIT 5 WHERE status != 'draft'")
        f = node.query_filter
        assert isinstance(f, CompareExpr)
        assert f.op == "!="
        assert f.value == "draft"

    def test_range_gt(self):
        node = parse("SEARCH docs SIMILAR TO 'ml' LIMIT 5 WHERE score > 0.8")
        f = node.query_filter
        assert isinstance(f, CompareExpr)
        assert f.op == ">"
        assert f.value == pytest.approx(0.8)

    def test_range_gte(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE year >= 2020")
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.op == ">="
        assert node.query_filter.value == 2020

    def test_range_lt(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE year < 2024")
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.op == "<"

    def test_range_lte(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE year <= 2023")
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.op == "<="

    def test_between(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE year BETWEEN 2018 AND 2023")
        f = node.query_filter
        assert isinstance(f, BetweenExpr)
        assert f.field == "year"
        assert f.low == 2018
        assert f.high == 2023

    def test_in_expr(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE status IN ('a', 'b')")
        f = node.query_filter
        assert isinstance(f, InExpr)
        assert f.field == "status"
        assert f.values == ("a", "b")

    def test_in_with_trailing_comma(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE status IN ('a', 'b',)")
        assert isinstance(node.query_filter, InExpr)
        assert len(node.query_filter.values) == 2

    def test_not_in_expr(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE status NOT IN ('deleted', 'archived')")
        f = node.query_filter
        assert isinstance(f, NotInExpr)
        assert f.values == ("deleted", "archived")

    def test_is_null(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE reviewer IS NULL")
        f = node.query_filter
        assert isinstance(f, IsNullExpr)
        assert f.field == "reviewer"

    def test_is_not_null(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE reviewer IS NOT NULL")
        assert isinstance(node.query_filter, IsNotNullExpr)
        assert node.query_filter.field == "reviewer"

    def test_is_empty(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE tags IS EMPTY")
        assert isinstance(node.query_filter, IsEmptyExpr)
        assert node.query_filter.field == "tags"

    def test_is_not_empty(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE tags IS NOT EMPTY")
        assert isinstance(node.query_filter, IsNotEmptyExpr)

    def test_match_text(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE title MATCH 'deep learning'")
        f = node.query_filter
        assert isinstance(f, MatchTextExpr)
        assert f.field == "title"
        assert f.text == "deep learning"

    def test_match_any(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE title MATCH ANY 'nlp ai'")
        f = node.query_filter
        assert isinstance(f, MatchAnyExpr)
        assert f.text == "nlp ai"

    def test_match_phrase(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE title MATCH PHRASE 'neural net'")
        assert isinstance(node.query_filter, MatchPhraseExpr)

    def test_and_expr_two_operands(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE a = '1' AND b = '2'")
        f = node.query_filter
        assert isinstance(f, AndExpr)
        assert len(f.operands) == 2
        assert all(isinstance(op, CompareExpr) for op in f.operands)

    def test_and_expr_three_operands_flattened(self):
        node = parse(
            "SEARCH d SIMILAR TO 'x' LIMIT 5 WHERE a = '1' AND b = '2' AND c = '3'"
        )
        f = node.query_filter
        assert isinstance(f, AndExpr)
        assert len(f.operands) == 3  # flattened, not binary-nested

    def test_or_expr(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE a = '1' OR b = '2'")
        f = node.query_filter
        assert isinstance(f, OrExpr)
        assert len(f.operands) == 2

    def test_not_expr(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE NOT status = 'draft'")
        f = node.query_filter
        assert isinstance(f, NotExpr)
        assert isinstance(f.operand, CompareExpr)

    def test_parenthesized_or_inside_and(self):
        node = parse(
            "SEARCH docs SIMILAR TO 'x' LIMIT 5 "
            "WHERE (src = 'a' OR src = 'b') AND year > 2020"
        )
        f = node.query_filter
        assert isinstance(f, AndExpr)
        assert isinstance(f.operands[0], OrExpr)
        assert isinstance(f.operands[1], CompareExpr)

    def test_dotted_field_path(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE meta.source = 'web'")
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.field == "meta.source"

    def test_using_model_then_where(self):
        node = parse(
            "SEARCH docs SIMILAR TO 'x' LIMIT 5 "
            "USING MODEL 'my-model' WHERE category = 'paper'"
        )
        assert node.model == "my-model"
        assert isinstance(node.query_filter, CompareExpr)

    def test_between_and_does_not_confuse_logical_and(self):
        # The AND inside BETWEEN must not be consumed by the logical AND loop
        node = parse(
            "SEARCH d SIMILAR TO 'x' LIMIT 5 WHERE year BETWEEN 2018 AND 2023 AND category = 'ai'"
        )
        f = node.query_filter
        assert isinstance(f, AndExpr)
        assert isinstance(f.operands[0], BetweenExpr)
        assert isinstance(f.operands[1], CompareExpr)
        assert len(f.operands) == 2

    def test_not_negates_parenthesized_group(self):
        node = parse(
            "SEARCH d SIMILAR TO 'x' LIMIT 5 WHERE NOT (a = '1' OR b = '2')"
        )
        f = node.query_filter
        assert isinstance(f, NotExpr)
        assert isinstance(f.operand, OrExpr)

    def test_missing_rparen_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE (a = '1'")


# ── Hybrid vector tests ───────────────────────────────────────────────────────

class TestHybridCreate:
    def test_create_hybrid_sets_flag(self):
        node = parse("CREATE COLLECTION articles HYBRID")
        assert isinstance(node, CreateCollectionStmt)
        assert node.collection == "articles"
        assert node.hybrid is True

    def test_create_non_hybrid_default_false(self):
        node = parse("CREATE COLLECTION articles")
        assert node.hybrid is False

    def test_create_hybrid_case_insensitive(self):
        node = parse("create collection col hybrid")
        assert node.hybrid is True


class TestCreateUsing:
    def test_create_using_model(self):
        node = parse("CREATE COLLECTION articles USING MODEL 'BAAI/bge-base-en-v1.5'")
        assert isinstance(node, CreateCollectionStmt)
        assert node.hybrid is False
        assert node.model == "BAAI/bge-base-en-v1.5"

    def test_create_using_hybrid(self):
        node = parse("CREATE COLLECTION articles USING HYBRID")
        assert isinstance(node, CreateCollectionStmt)
        assert node.hybrid is True
        assert node.model is None

    def test_create_using_hybrid_dense_model(self):
        node = parse("CREATE COLLECTION articles USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'")
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"

    def test_create_bare_hybrid_backward_compat(self):
        node = parse("CREATE COLLECTION articles HYBRID")
        assert node.hybrid is True
        assert node.model is None

    def test_create_plain_backward_compat(self):
        node = parse("CREATE COLLECTION articles")
        assert node.hybrid is False
        assert node.model is None

    def test_create_using_model_sets_collection_name(self):
        node = parse("CREATE COLLECTION my_col USING MODEL 'some/model'")
        assert isinstance(node, CreateCollectionStmt)
        assert node.collection == "my_col"

    def test_create_using_hybrid_case_insensitive(self):
        node = parse("create collection articles using hybrid")
        assert node.hybrid is True

    def test_create_using_model_case_insensitive(self):
        node = parse("create collection articles using model 'some/model'")
        assert node.model == "some/model"


class TestHybridInsert:
    def test_insert_using_hybrid_sets_flag(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'hi'} USING HYBRID")
        assert isinstance(node, InsertStmt)
        assert node.hybrid is True
        assert node.model is None
        assert node.sparse_model is None

    def test_insert_non_hybrid_default(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'hi'}")
        assert node.hybrid is False
        assert node.sparse_model is None

    def test_insert_using_model_still_works(self):
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'hi'} USING MODEL 'my-model'")
        assert node.hybrid is False
        assert node.model == "my-model"
        assert node.sparse_model is None

    def test_insert_hybrid_dense_model(self):
        node = parse(
            "INSERT INTO COLLECTION col VALUES {'text': 'hi'} "
            "USING HYBRID DENSE MODEL 'BAAI/bge-small-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-small-en-v1.5"
        assert node.sparse_model is None

    def test_insert_hybrid_sparse_model(self):
        node = parse(
            "INSERT INTO COLLECTION col VALUES {'text': 'hi'} "
            "USING HYBRID SPARSE MODEL 'Qdrant/bm25'"
        )
        assert node.hybrid is True
        assert node.model is None
        assert node.sparse_model == "Qdrant/bm25"

    def test_insert_hybrid_both_models(self):
        node = parse(
            "INSERT INTO COLLECTION col VALUES {'text': 'hi'} "
            "USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'Qdrant/bm25'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.sparse_model == "Qdrant/bm25"

    def test_insert_hybrid_both_models_reversed_order(self):
        node = parse(
            "INSERT INTO COLLECTION col VALUES {'text': 'hi'} "
            "USING HYBRID SPARSE MODEL 'Qdrant/bm25' DENSE MODEL 'BAAI/bge-base-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.sparse_model == "Qdrant/bm25"


class TestHybridSearch:
    def test_search_using_hybrid_sets_flag(self):
        node = parse("SEARCH articles SIMILAR TO 'ml' LIMIT 10 USING HYBRID")
        assert isinstance(node, SearchStmt)
        assert node.hybrid is True
        assert node.model is None
        assert node.sparse_model is None

    def test_search_non_hybrid_default(self):
        node = parse("SEARCH articles SIMILAR TO 'ml' LIMIT 10")
        assert node.hybrid is False
        assert node.sparse_model is None

    def test_search_using_model_still_works(self):
        node = parse("SEARCH articles SIMILAR TO 'ml' LIMIT 5 USING MODEL 'my-model'")
        assert node.hybrid is False
        assert node.model == "my-model"
        assert node.sparse_model is None

    def test_search_hybrid_dense_model(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID DENSE MODEL 'BAAI/bge-small-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-small-en-v1.5"
        assert node.sparse_model is None

    def test_search_hybrid_sparse_model(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID SPARSE MODEL 'prithivida/Splade_PP_en_v1'"
        )
        assert node.hybrid is True
        assert node.model is None
        assert node.sparse_model == "prithivida/Splade_PP_en_v1"

    def test_search_hybrid_both_models(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'Qdrant/bm25'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.sparse_model == "Qdrant/bm25"

    def test_search_hybrid_both_models_reversed_order(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID SPARSE MODEL 'Qdrant/bm25' DENSE MODEL 'BAAI/bge-base-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.sparse_model == "Qdrant/bm25"

    def test_search_hybrid_with_where(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 USING HYBRID WHERE year > 2020"
        )
        assert node.hybrid is True
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.field == "year"

    def test_search_hybrid_dense_model_and_where(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID DENSE MODEL 'BAAI/bge-small-en-v1.5' WHERE year > 2020"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-small-en-v1.5"
        assert isinstance(node.query_filter, CompareExpr)

    def test_search_hybrid_limit_preserved(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 7 USING HYBRID")
        assert node.limit == 7


class TestRerankSearch:
    def test_rerank_flag_set(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 RERANK")
        assert node.rerank is True
        assert node.rerank_model is None

    def test_rerank_with_model(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 RERANK MODEL 'cross-encoder/ms-marco-MiniLM-L-6-v2'"
        )
        assert node.rerank is True
        assert node.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_rerank_default_false(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5")
        assert node.rerank is False
        assert node.rerank_model is None

    def test_rerank_with_using_model(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING MODEL 'BAAI/bge-small-en-v1.5' RERANK")
        assert node.model == "BAAI/bge-small-en-v1.5"
        assert node.rerank is True

    def test_rerank_with_hybrid(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID RERANK")
        assert node.hybrid is True
        assert node.rerank is True
        assert node.rerank_model is None

    def test_rerank_with_where(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WHERE year > 2020 RERANK")
        assert node.query_filter is not None
        assert node.rerank is True

    def test_rerank_with_hybrid_where_and_model(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID WHERE year > 2020 "
            "RERANK MODEL 'cross-encoder/ms-marco-MiniLM-L-6-v2'"
        )
        assert node.hybrid is True
        assert node.query_filter is not None
        assert node.rerank is True
        assert node.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_rerank_lowercase(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 rerank")
        assert node.rerank is True

    def test_rerank_model_custom(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 RERANK MODEL 'my-custom/reranker'")
        assert node.rerank_model == "my-custom/reranker"

    def test_existing_search_unaffected_by_rerank_addition(self):
        """Existing parse calls without RERANK still produce rerank=False."""
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 10 USING MODEL 'BAAI/bge-small-en-v1.5'")
        assert node.rerank is False
        assert node.rerank_model is None


class TestExactSearch:
    def test_exact_keyword_sets_flag(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 EXACT")
        assert node.with_clause is not None

    def test_exact_with_where(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 EXACT WHERE year > 2020")
        assert node.with_clause is not None
        assert node.query_filter is not None

    def test_exact_with_hybrid(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID EXACT")
        assert node.hybrid is True
        assert node.with_clause is not None


class TestSearchWithClause:
    def test_with_hnsw_ef(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { hnsw_ef: 256 }")
        assert node.with_clause is not None
        assert node.with_clause.hnsw_ef == 256

    def test_with_exact_true(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { exact: true }")
        assert node.with_clause is not None
        assert node.with_clause.exact is True

    def test_with_exact_false(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { exact: false }")
        assert node.with_clause is not None
        assert node.with_clause.exact is False

    def test_with_acorn(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { acorn: true }")
        assert node.with_clause is not None
        assert node.with_clause.acorn is True

    def test_with_multiple_params(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { hnsw_ef: 256, acorn: true }"
        )
        assert node.with_clause.hnsw_ef == 256
        assert node.with_clause.acorn is True

    def test_with_after_where(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 WHERE year > 2020 WITH { hnsw_ef: 128 }"
        )
        assert node.with_clause is not None
        assert node.with_clause.hnsw_ef == 128
        assert node.query_filter is not None

    def test_with_after_rerank(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 RERANK WITH { hnsw_ef: 256 }")
        assert node.rerank is True
        assert node.with_clause is not None
        assert node.with_clause.hnsw_ef == 256

    def test_with_full_search(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID WHERE year > 2020 "
            "RERANK WITH { hnsw_ef: 256, acorn: true }"
        )
        assert node.hybrid is True
        assert node.query_filter is not None
        assert node.rerank is True
        assert node.with_clause is not None
        assert node.with_clause.hnsw_ef == 256
        assert node.with_clause.acorn is True

    def test_with_unknown_keyword_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { diversity: 0.5 }")

    def test_with_trailing_comma(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { hnsw_ef: 256, }")
        assert node.with_clause.hnsw_ef == 256


class TestSparseOnlySearch:
    def test_using_sparse_sets_flag(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING SPARSE")
        assert node.sparse_only is True
        assert node.hybrid is False
        assert node.sparse_model is None

    def test_using_sparse_with_model(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 USING SPARSE MODEL 'prithivida/Splade_PP_en_v1'"
        )
        assert node.sparse_only is True
        assert node.sparse_model == "prithivida/Splade_PP_en_v1"

    def test_using_sparse_default_flags(self):
        """All other fields remain at their defaults when USING SPARSE is used."""
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING SPARSE")
        assert node.hybrid is False
        assert node.model is None
        assert node.rerank is False
        assert node.query_filter is None

    def test_using_sparse_with_where(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 USING SPARSE WHERE year > 2020"
        )
        assert node.sparse_only is True
        assert node.query_filter is not None

    def test_using_sparse_with_rerank(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING SPARSE RERANK")
        assert node.sparse_only is True
        assert node.rerank is True

    def test_using_sparse_with_model_and_rerank(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 "
            "USING SPARSE MODEL 'prithivida/Splade_PP_en_v1' RERANK"
        )
        assert node.sparse_only is True
        assert node.sparse_model == "prithivida/Splade_PP_en_v1"
        assert node.rerank is True

    def test_sparse_only_false_by_default(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5")
        assert node.sparse_only is False

    def test_sparse_only_false_for_hybrid(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID")
        assert node.sparse_only is False
        assert node.hybrid is True
