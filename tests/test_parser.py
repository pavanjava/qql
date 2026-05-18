import pytest

from qql.ast_nodes import (
    AlterCollectionStmt,
    AndExpr,
    BetweenExpr,
    CollectionParamsConfig,
    CollectionConfig,
    CompareExpr,
    CreateCollectionStmt,
    CreateIndexStmt,
    DeleteStmt,
    DropCollectionStmt,
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
    OptimizersRuntimeConfig,
    OrExpr,
    QuantizationUpdate,
    QuantizationType,
    QuantizationSearchWith,
    RecommendStmt,
    SelectStmt,
    ScrollStmt,
    SearchStmt,
    ShowCollectionStmt,
    ShowCollectionsStmt,
    VectorsConfig,
    HnswRuntimeConfig,
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


class TestInsertBulk:
    def test_basic_bulk_insert(self):
        node = parse("INSERT BULK INTO COLLECTION col VALUES [{'text': 'hello'}]")
        assert isinstance(node, InsertBulkStmt)
        assert node.collection == "col"
        assert len(node.values_list) == 1
        assert node.values_list[0]["text"] == "hello"

    def test_bulk_insert_two_items(self):
        node = parse(
            "INSERT BULK INTO COLLECTION col VALUES "
            "[{'text': 'first'}, {'text': 'second'}]"
        )
        assert isinstance(node, InsertBulkStmt)
        assert len(node.values_list) == 2
        assert node.values_list[1]["text"] == "second"

    def test_bulk_insert_preserves_metadata(self):
        node = parse(
            "INSERT BULK INTO COLLECTION col VALUES "
            "[{'text': 'hello', 'year': 2021}, {'text': 'world', 'year': 2022}]"
        )
        assert node.values_list[0]["year"] == 2021
        assert node.values_list[1]["year"] == 2022

    def test_bulk_insert_using_model(self):
        node = parse(
            "INSERT BULK INTO COLLECTION col VALUES [{'text': 'a'}] "
            "USING MODEL 'BAAI/bge-base-en-v1.5'"
        )
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.hybrid is False

    def test_bulk_insert_using_hybrid(self):
        node = parse(
            "INSERT BULK INTO COLLECTION col VALUES [{'text': 'a'}] USING HYBRID"
        )
        assert node.hybrid is True
        assert node.model is None

    def test_bulk_insert_using_hybrid_dense_model(self):
        node = parse(
            "INSERT BULK INTO COLLECTION col VALUES [{'text': 'a'}] "
            "USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"

    def test_bulk_insert_collection_name(self):
        node = parse("INSERT BULK INTO COLLECTION my_notes VALUES [{'text': 'x'}]")
        assert node.collection == "my_notes"

    def test_bulk_insert_case_insensitive(self):
        node = parse("insert bulk into collection col values [{'text': 'hi'}]")
        assert isinstance(node, InsertBulkStmt)

    def test_bulk_insert_default_model_is_none(self):
        node = parse("INSERT BULK INTO COLLECTION col VALUES [{'text': 'a'}]")
        assert node.model is None
        assert node.sparse_model is None
        assert node.hybrid is False

    def test_single_insert_still_works_after_bulk_addition(self):
        """Ensure single INSERT flow is not broken by the BULK branch."""
        node = parse("INSERT INTO COLLECTION col VALUES {'text': 'hello'}")
        assert isinstance(node, InsertStmt)
        assert node.values == {"text": "hello"}


class TestCreate:
    def test_create_collection(self):
        node = parse("CREATE COLLECTION my_col")
        assert isinstance(node, CreateCollectionStmt)
        assert node.collection == "my_col"

    def test_create_index(self):
        node = parse("CREATE INDEX ON COLLECTION articles FOR category TYPE keyword")
        assert isinstance(node, CreateIndexStmt)
        assert node.collection == "articles"
        assert node.field_name == "category"
        assert node.schema == "keyword"


class TestDrop:
    def test_drop_collection(self):
        node = parse("DROP COLLECTION my_col")
        assert isinstance(node, DropCollectionStmt)
        assert node.collection == "my_col"


class TestShow:
    def test_show_collections(self):
        node = parse("SHOW COLLECTIONS")
        assert isinstance(node, ShowCollectionsStmt)

    def test_show_collection(self):
        node = parse("SHOW COLLECTION docs")
        assert isinstance(node, ShowCollectionStmt)
        assert node.collection == "docs"

    def test_show_collection_case_insensitive(self):
        node = parse("show collection MY_COL")
        assert isinstance(node, ShowCollectionStmt)
        assert node.collection == "MY_COL"


class TestScroll:
    def test_scroll_basic(self):
        node = parse("SCROLL FROM docs LIMIT 50")
        assert isinstance(node, ScrollStmt)
        assert node.collection == "docs"
        assert node.limit == 50
        assert node.query_filter is None
        assert node.after is None

    def test_scroll_with_where(self):
        node = parse("SCROLL FROM docs WHERE year >= 2024 LIMIT 50")
        assert isinstance(node, ScrollStmt)
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.field == "year"
        assert node.after is None

    def test_scroll_with_after(self):
        node = parse("SCROLL FROM docs AFTER 'cursor-id' LIMIT 50")
        assert isinstance(node, ScrollStmt)
        assert node.after == "cursor-id"

    def test_scroll_with_where_and_after(self):
        node = parse("SCROLL FROM docs WHERE year >= 2024 AFTER 42 LIMIT 50")
        assert isinstance(node, ScrollStmt)
        assert node.after == 42
        assert isinstance(node.query_filter, CompareExpr)


class TestSelect:
    def test_select_by_string_id(self):
        node = parse("SELECT * FROM notes WHERE id = 'abc-123'")
        assert isinstance(node, SelectStmt)
        assert node.collection == "notes"
        assert node.point_id == "abc-123"

    def test_select_by_integer_id(self):
        node = parse("SELECT * FROM notes WHERE id = 42")
        assert isinstance(node, SelectStmt)
        assert node.point_id == 42

    def test_select_requires_id_filter(self):
        with pytest.raises(QQLSyntaxError):
            parse("SELECT * FROM notes WHERE year = 2024")


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
        assert node.query_filter is None

    def test_delete_by_integer_id(self):
        node = parse("DELETE FROM notes WHERE id = 99")
        assert isinstance(node, DeleteStmt)
        assert node.point_id == 99

    def test_delete_by_filter(self):
        node = parse("DELETE FROM articles WHERE category = 'archived'")
        assert isinstance(node, DeleteStmt)
        assert node.point_id is None
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.field == "category"
        assert node.query_filter.value == "archived"


class TestRecommend:
    def test_recommend_with_positive_ids(self):
        node = parse("RECOMMEND FROM notes POSITIVE IDS ('a', 'b') LIMIT 5")
        assert isinstance(node, RecommendStmt)
        assert node.collection == "notes"
        assert node.positive_ids == ("a", "b")
        assert node.negative_ids == ()
        assert node.limit == 5
        assert node.strategy is None

    def test_recommend_with_negative_ids_and_strategy(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a', 2) "
            "NEGATIVE IDS ('x') STRATEGY 'best_score' LIMIT 7"
        )
        assert node.positive_ids == ("a", 2)
        assert node.negative_ids == ("x",)
        assert node.strategy == "best_score"
        assert node.limit == 7

    def test_recommend_with_where_filter(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 5 WHERE year > 2020"
        )
        assert isinstance(node.query_filter, CompareExpr)
        assert node.query_filter.field == "year"

    def test_recommend_requires_non_empty_positive_ids(self):
        with pytest.raises(QQLSyntaxError):
            parse("RECOMMEND FROM notes POSITIVE IDS () LIMIT 5")

    def test_recommend_with_offset(self):
        node = parse("RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 10 OFFSET 5")
        assert node.offset == 5

    def test_recommend_with_score_threshold(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 10 SCORE THRESHOLD 0.5"
        )
        assert node.score_threshold == pytest.approx(0.5)

    def test_recommend_with_clause(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 10 WITH { exact: true }"
        )
        assert node.with_clause is not None
        assert node.with_clause.exact is True

    def test_recommend_with_clause_hnsw_ef(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 10 WITH { hnsw_ef: 128 }"
        )
        assert node.with_clause is not None
        assert node.with_clause.hnsw_ef == 128

    def test_recommend_with_indexed_only_and_quantization(self):
        node = parse(
            "RECOMMEND FROM notes POSITIVE IDS ('a') LIMIT 10 "
            "WITH { indexed_only: true, quantization: { rescore: true } }"
        )
        assert node.with_clause is not None
        assert node.with_clause.indexed_only is True
        assert node.with_clause.quantization is not None
        assert node.with_clause.quantization.rescore is True

    def test_recommend_lookup_from(self):
        node = parse(
            "RECOMMEND FROM target_collection POSITIVE IDS ('a') "
            "LOOKUP FROM source_collection LIMIT 5"
        )
        assert node.lookup_from == ("source_collection", None)

    def test_recommend_lookup_from_with_vector(self):
        node = parse(
            "RECOMMEND FROM target_collection POSITIVE IDS ('a') "
            "LOOKUP FROM source_collection VECTOR 'dense' LIMIT 5"
        )
        assert node.lookup_from == ("source_collection", "dense")

    def test_recommend_using(self):
        node = parse(
            "RECOMMEND FROM docs POSITIVE IDS ('a') USING 'sparse' LIMIT 5"
        )
        assert node.using == "sparse"

    def test_recommend_lookup_from_and_using(self):
        node = parse(
            "RECOMMEND FROM target_collection POSITIVE IDS ('a') "
            "LOOKUP FROM source_collection VECTOR 'dense' USING 'sparse' LIMIT 5"
        )
        assert node.lookup_from == ("source_collection", "dense")
        assert node.using == "sparse"

    def test_recommend_full_clause_order(self):
        node = parse(
            "RECOMMEND FROM docs POSITIVE IDS ('a', 'b') "
            "NEGATIVE IDS ('x') STRATEGY 'best_score' "
            "LOOKUP FROM src VECTOR 'dense' USING 'sparse' "
            "LIMIT 10 OFFSET 5 SCORE THRESHOLD 0.5 "
            "WHERE year > 2020 WITH { exact: true, hnsw_ef: 128 }"
        )
        assert node.collection == "docs"
        assert node.positive_ids == ("a", "b")
        assert node.negative_ids == ("x",)
        assert node.strategy == "best_score"
        assert node.lookup_from == ("src", "dense")
        assert node.using == "sparse"
        assert node.limit == 10
        assert node.offset == 5
        assert node.score_threshold == pytest.approx(0.5)
        assert isinstance(node.query_filter, CompareExpr)
        assert node.with_clause is not None
        assert node.with_clause.exact is True
        assert node.with_clause.hnsw_ef == 128


class TestErrors:
    def test_unknown_keyword(self):
        with pytest.raises(QQLSyntaxError):
            parse("UPSERT INTO foo VALUES {'text': 'x'}")

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

    def test_boolean_equality_filter(self):
        node = parse("SEARCH docs SIMILAR TO 'ml' LIMIT 5 WHERE active = true")
        f = node.query_filter
        assert isinstance(f, CompareExpr)
        assert f.field == "active"
        assert f.op == "="
        assert f.value is True

    def test_boolean_in_expr(self):
        node = parse("SEARCH docs SIMILAR TO 'x' LIMIT 5 WHERE active IN (true, false)")
        f = node.query_filter
        assert isinstance(f, InExpr)
        assert f.values == (True, False)

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


class TestCollectionConfig:
    def test_create_with_collection_blocks(self):
        node = parse(
            "CREATE COLLECTION articles "
            "WITH VECTORS { on_disk: true } "
            "WITH HNSW { m: 32, ef_construct: 200, full_scan_threshold: 5000, "
            "max_indexing_threads: 2, on_disk: true, payload_m: 24, inline_storage: false } "
            "WITH OPTIMIZERS { indexing_threshold: 10000, memmap_threshold: 20000, deleted_threshold: 0.2, max_optimization_threads: 'auto' } "
            "WITH PARAMS { replication_factor: 2, write_consistency_factor: 1, on_disk_payload: true } "
            "QUANTIZE SCALAR ALWAYS RAM"
        )
        assert isinstance(node, CreateCollectionStmt)
        assert node.config == CollectionConfig(
            vectors=VectorsConfig(on_disk=True),
            hnsw=HnswRuntimeConfig(
                m=32,
                ef_construct=200,
                full_scan_threshold=5000,
                max_indexing_threads=2,
                on_disk=True,
                payload_m=24,
                inline_storage=False,
            ),
            optimizers=OptimizersRuntimeConfig(
                indexing_threshold=10000,
                memmap_threshold=20000,
                deleted_threshold=0.2,
                max_optimization_threads="auto",
            ),
            params=CollectionParamsConfig(
                replication_factor=2,
                write_consistency_factor=1,
                on_disk_payload=True,
            ),
        )
        assert node.quantization is not None
        assert node.quantization.type == QuantizationType.SCALAR
        assert node.quantization.always_ram is True

    def test_alter_with_collection_blocks(self):
        node = parse(
            "ALTER COLLECTION articles "
            "WITH VECTORS { on_disk: true } "
            "WITH HNSW { full_scan_threshold: 1234 } "
            "WITH OPTIMIZERS { indexing_threshold: 4567, prevent_unoptimized: true } "
            "WITH PARAMS { on_disk_payload: false, read_fan_out_factor: 4 } "
            "QUANTIZE DISABLED"
        )
        assert isinstance(node, AlterCollectionStmt)
        assert node.config == CollectionConfig(
            vectors=VectorsConfig(on_disk=True),
            hnsw=HnswRuntimeConfig(full_scan_threshold=1234),
            optimizers=OptimizersRuntimeConfig(
                indexing_threshold=4567,
                prevent_unoptimized=True,
            ),
            params=CollectionParamsConfig(
                on_disk_payload=False,
                read_fan_out_factor=4,
            ),
        )
        assert node.quantization == QuantizationUpdate(disabled=True)

    def test_create_hnsw_without_with_rejected(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles HNSW { payload_m: 24 }")

    def test_duplicate_collection_config_rejected(self):
        with pytest.raises(QQLSyntaxError):
            parse(
                "CREATE COLLECTION articles "
                "WITH HNSW { payload_m: 24 } "
                "WITH HNSW { payload_m: 32 }"
            )

    def test_create_quantize_must_come_after_with_blocks(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE SCALAR WITH HNSW { payload_m: 24 }")

    def test_create_rejects_alter_only_params(self):
        with pytest.raises(QQLSyntaxError, match="supported only for ALTER COLLECTION"):
            parse(
                "CREATE COLLECTION articles "
                "WITH PARAMS { read_fan_out_factor: 4 }"
            )


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

    def test_search_hybrid_with_dbsf_fusion(self):
        node = parse(
            "SEARCH docs SIMILAR TO 'q' LIMIT 10 USING HYBRID FUSION 'dbsf'"
        )
        assert node.hybrid is True
        assert node.fusion == "dbsf"

    def test_search_hybrid_with_fusion_and_models(self):
        node = parse(
            "SEARCH docs SIMILAR TO 'q' LIMIT 10 "
            "USING HYBRID FUSION 'rrf' SPARSE MODEL 'Qdrant/bm25' "
            "DENSE MODEL 'BAAI/bge-base-en-v1.5'"
        )
        assert node.hybrid is True
        assert node.fusion == "rrf"
        assert node.sparse_model == "Qdrant/bm25"
        assert node.model == "BAAI/bge-base-en-v1.5"

    def test_search_hybrid_dense_model_and_where(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'ml' LIMIT 10 "
            "USING HYBRID DENSE MODEL 'BAAI/bge-small-en-v1.5' WHERE year > 2020"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-small-en-v1.5"
        assert isinstance(node.query_filter, CompareExpr)

    def test_search_hybrid_rejects_unknown_fusion(self):
        with pytest.raises(QQLSyntaxError, match="Unsupported hybrid fusion"):
            parse("SEARCH docs SIMILAR TO 'q' LIMIT 10 USING HYBRID FUSION 'x'")

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

    def test_with_indexed_only(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { indexed_only: true }")
        assert node.with_clause is not None
        assert node.with_clause.indexed_only is True

    def test_with_quantization(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 "
            "WITH { quantization: { ignore: true, rescore: false, oversampling: 2 } }"
        )
        assert node.with_clause is not None
        assert node.with_clause.quantization is not None
        assert node.with_clause.quantization.ignore is True
        assert node.with_clause.quantization.rescore is False
        assert node.with_clause.quantization.oversampling == pytest.approx(2.0)

    def test_with_multiple_params(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { hnsw_ef: 256, acorn: true }"
        )
        assert node.with_clause.hnsw_ef == 256
        assert node.with_clause.acorn is True

    def test_with_mmr_params(self):
        node = parse(
            "SEARCH col SIMILAR TO 'q' LIMIT 5 "
            "WITH { mmr_diversity: 0.5, mmr_candidates: 50 }"
        )
        assert node.with_clause is not None
        assert node.with_clause.mmr_diversity == pytest.approx(0.5)
        assert node.with_clause.mmr_candidates == 50

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

    def test_with_mmr_diversity_out_of_range_raises(self):
        with pytest.raises(QQLSyntaxError, match="mmr_diversity must be between 0 and 1"):
            parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { mmr_diversity: 1.5 }")

    def test_with_mmr_candidates_non_positive_raises(self):
        with pytest.raises(QQLSyntaxError, match="mmr_candidates must be a positive integer"):
            parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { mmr_candidates: 0 }")

    def test_with_quantization_unknown_key_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { quantization: { unknown: true } }")

    def test_with_trailing_comma(self):
        node = parse("SEARCH col SIMILAR TO 'q' LIMIT 5 WITH { hnsw_ef: 256, }")
        assert node.with_clause.hnsw_ef == 256

    def test_with_quantization_unknown_key_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse(
                "SEARCH col SIMILAR TO 'q' LIMIT 5 "
                "WITH { quantization: { unknown: true } }"
            )


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


# ── TestQuantizeCreate ────────────────────────────────────────────────────────


class TestQuantizeCreate:
    # ── Scalar — no options ───────────────────────────────────────────────

    def test_scalar_no_options(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR")
        assert isinstance(node, CreateCollectionStmt)
        assert node.quantization is not None
        assert node.quantization.type == QuantizationType.SCALAR
        assert node.quantization.quantile is None
        assert node.quantization.always_ram is False

    def test_scalar_with_quantile(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 0.99")
        assert node.quantization.type == QuantizationType.SCALAR
        assert node.quantization.quantile == pytest.approx(0.99)

    def test_scalar_with_quantile_zero(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 0")
        assert node.quantization.type == QuantizationType.SCALAR
        assert node.quantization.quantile == pytest.approx(0.0)

    def test_scalar_with_quantile_one(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 1")
        assert node.quantization.type == QuantizationType.SCALAR
        assert node.quantization.quantile == pytest.approx(1.0)

    def test_scalar_with_always_ram(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR ALWAYS RAM")
        assert node.quantization.always_ram is True
        assert node.quantization.quantile is None

    def test_scalar_quantile_and_always_ram(self):
        node = parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 0.95 ALWAYS RAM")
        assert node.quantization.quantile == pytest.approx(0.95)
        assert node.quantization.always_ram is True

    # ── Binary ────────────────────────────────────────────────────────────

    def test_binary_no_options(self):
        node = parse("CREATE COLLECTION articles QUANTIZE BINARY")
        assert isinstance(node, CreateCollectionStmt)
        assert node.quantization.type == QuantizationType.BINARY
        assert node.quantization.always_ram is False

    def test_binary_with_always_ram(self):
        node = parse("CREATE COLLECTION articles QUANTIZE BINARY ALWAYS RAM")
        assert node.quantization.type == QuantizationType.BINARY
        assert node.quantization.always_ram is True

    # ── Product ───────────────────────────────────────────────────────────

    def test_product_no_options(self):
        node = parse("CREATE COLLECTION articles QUANTIZE PRODUCT")
        assert isinstance(node, CreateCollectionStmt)
        assert node.quantization.type == QuantizationType.PRODUCT
        assert node.quantization.always_ram is False

    def test_product_with_always_ram(self):
        node = parse("CREATE COLLECTION articles QUANTIZE PRODUCT ALWAYS RAM")
        assert node.quantization.type == QuantizationType.PRODUCT
        assert node.quantization.always_ram is True

    # ── Combined with HYBRID / MODEL ──────────────────────────────────────

    def test_combined_with_hybrid_shorthand(self):
        node = parse("CREATE COLLECTION articles HYBRID QUANTIZE SCALAR")
        assert node.hybrid is True
        assert node.quantization.type == QuantizationType.SCALAR

    def test_combined_with_using_hybrid(self):
        node = parse("CREATE COLLECTION articles USING HYBRID QUANTIZE BINARY")
        assert node.hybrid is True
        assert node.quantization.type == QuantizationType.BINARY

    def test_combined_with_using_model(self):
        node = parse(
            "CREATE COLLECTION articles USING MODEL 'BAAI/bge-base-en-v1.5' QUANTIZE SCALAR"
        )
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.hybrid is False
        assert node.quantization.type == QuantizationType.SCALAR

    def test_combined_with_hybrid_dense_model(self):
        node = parse(
            "CREATE COLLECTION articles USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'"
            " QUANTIZE SCALAR"
        )
        assert node.hybrid is True
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.quantization.type == QuantizationType.SCALAR

    # ── Backward compatibility ────────────────────────────────────────────

    def test_no_quantize_clause_is_none(self):
        node = parse("CREATE COLLECTION articles")
        assert node.quantization is None

    def test_no_quantize_with_hybrid_is_none(self):
        node = parse("CREATE COLLECTION articles HYBRID")
        assert node.hybrid is True
        assert node.quantization is None

    # ── Error cases ───────────────────────────────────────────────────────

    def test_quantize_missing_type_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE")

    def test_quantize_unknown_type_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE FULL")

    def test_scalar_quantile_above_one_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 1.5")

    def test_scalar_quantile_integer_above_one_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE SCALAR QUANTILE 2")


class TestTurboQuantCreate:
    """Parser tests for QUANTIZE TURBO [BITS n] [ALWAYS RAM]."""

    # ── Default / no options ──────────────────────────────────────────────

    def test_turbo_no_options(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO")
        assert node.quantization is not None
        assert node.quantization.type == QuantizationType.TURBO
        assert node.quantization.turbo_bits is None
        assert node.quantization.always_ram is False

    # ── BITS variants ─────────────────────────────────────────────────────

    def test_turbo_bits4(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 4")
        assert node.quantization.type == QuantizationType.TURBO
        assert node.quantization.turbo_bits == 4.0

    def test_turbo_bits2(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 2")
        assert node.quantization.turbo_bits == 2.0

    def test_turbo_bits1_5(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 1.5")
        assert node.quantization.turbo_bits == 1.5

    def test_turbo_bits1(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 1")
        assert node.quantization.turbo_bits == 1.0

    # ── ALWAYS RAM ────────────────────────────────────────────────────────

    def test_turbo_always_ram_no_bits(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO ALWAYS RAM")
        assert node.quantization.type == QuantizationType.TURBO
        assert node.quantization.always_ram is True
        assert node.quantization.turbo_bits is None

    def test_turbo_bits_and_always_ram(self):
        node = parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 2 ALWAYS RAM")
        assert node.quantization.turbo_bits == 2.0
        assert node.quantization.always_ram is True

    # ── Composed with other clauses ───────────────────────────────────────

    def test_turbo_with_hybrid_shorthand(self):
        node = parse("CREATE COLLECTION articles HYBRID QUANTIZE TURBO")
        assert node.hybrid is True
        assert node.quantization.type == QuantizationType.TURBO

    def test_turbo_with_using_hybrid(self):
        node = parse("CREATE COLLECTION articles USING HYBRID QUANTIZE TURBO BITS 2")
        assert node.hybrid is True
        assert node.quantization.turbo_bits == 2.0

    def test_turbo_with_model(self):
        node = parse("CREATE COLLECTION articles USING MODEL 'BAAI/bge-base-en-v1.5' QUANTIZE TURBO BITS 1.5")
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.quantization.type == QuantizationType.TURBO
        assert node.quantization.turbo_bits == 1.5

    def test_turbo_with_hybrid_dense_model(self):
        node = parse("CREATE COLLECTION articles USING HYBRID DENSE MODEL 'x' QUANTIZE TURBO BITS 1 ALWAYS RAM")
        assert node.hybrid is True
        assert node.model == "x"
        assert node.quantization.turbo_bits == 1.0
        assert node.quantization.always_ram is True

    # ── Error cases ───────────────────────────────────────────────────────

    def test_turbo_invalid_bits_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 3")

    def test_turbo_invalid_bits_float_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("CREATE COLLECTION articles QUANTIZE TURBO BITS 0.5")


# ── New feature tests ─────────────────────────────────────────────────────────

class TestSearchGroupBy:
    def test_group_by_basic(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 GROUP BY category")
        assert isinstance(node, SearchStmt)
        assert node.group_by == "category"
        assert node.group_size == 3  # default

    def test_group_by_with_group_size(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 GROUP BY category GROUP_SIZE 5")
        assert node.group_by == "category"
        assert node.group_size == 5

    def test_group_by_with_where(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 WHERE year >= 2020 GROUP BY category")
        assert node.group_by == "category"
        assert node.query_filter is not None

    def test_group_by_with_where_and_group_size(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'query' LIMIT 5 WHERE year >= 2020 "
            "GROUP BY category GROUP_SIZE 2"
        )
        assert node.group_by == "category"
        assert node.group_size == 2
        assert node.query_filter is not None

    def test_group_by_with_hybrid(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 USING HYBRID GROUP BY category")
        assert node.hybrid is True
        assert node.group_by == "category"

    def test_group_by_dotted_field(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 GROUP BY meta.author")
        assert node.group_by == "meta.author"

    def test_group_by_rerank_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 RERANK GROUP BY category")

    def test_plain_search_has_no_group_by(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 10")
        assert node.group_by is None

    def test_group_size_default_is_3(self):
        node = parse("SEARCH articles SIMILAR TO 'query' LIMIT 5 GROUP BY tag")
        assert node.group_size == 3

    def test_group_by_with_model(self):
        node = parse(
            "SEARCH articles SIMILAR TO 'query' LIMIT 5 "
            "USING MODEL 'BAAI/bge-base-en-v1.5' GROUP BY category"
        )
        assert node.model == "BAAI/bge-base-en-v1.5"
        assert node.group_by == "category"

    def test_group_by_collection_stored(self):
        node = parse("SEARCH notes SIMILAR TO 'query' LIMIT 3 GROUP BY topic GROUP_SIZE 4")
        assert node.collection == "notes"
        assert node.limit == 3
        assert node.group_by == "topic"
        assert node.group_size == 4


class TestUpdateVector:
    def test_update_vector_by_string_id(self):
        from qql.ast_nodes import UpdateVectorStmt
        node = parse("UPDATE articles SET VECTOR WHERE id = 'abc-123' [0.1, 0.2, 0.3]")
        assert isinstance(node, UpdateVectorStmt)
        assert node.collection == "articles"
        assert node.point_id == "abc-123"
        assert node.vector == (0.1, 0.2, 0.3)

    def test_update_vector_by_integer_id(self):
        from qql.ast_nodes import UpdateVectorStmt
        node = parse("UPDATE articles SET VECTOR WHERE id = 42 [0.1, 0.2]")
        assert isinstance(node, UpdateVectorStmt)
        assert node.point_id == 42

    def test_update_vector_parses_float_list(self):
        from qql.ast_nodes import UpdateVectorStmt
        node = parse("UPDATE notes SET VECTOR WHERE id = 1 [0.1, 0.2, 0.3, 0.4]")
        assert isinstance(node, UpdateVectorStmt)
        assert len(node.vector) == 4
        assert all(isinstance(v, float) for v in node.vector)

    def test_update_vector_collection_stored(self):
        node = parse("UPDATE my_col SET VECTOR WHERE id = 99 [0.5]")
        assert node.collection == "my_col"

    def test_update_vector_wrong_keyword_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("UPDATE articles SET FOOBAR WHERE id = 1 [0.1]")

    def test_update_vector_missing_brackets_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("UPDATE articles SET VECTOR WHERE id = 1 0.1 0.2")

    def test_update_vector_missing_id_eq_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("UPDATE articles SET VECTOR WHERE 'abc' [0.1]")

    def test_update_vector_large_vector(self):
        from qql.ast_nodes import UpdateVectorStmt
        vec = ", ".join(["0.1"] * 384)
        node = parse(f"UPDATE articles SET VECTOR WHERE id = 1 [{vec}]")
        assert isinstance(node, UpdateVectorStmt)
        assert len(node.vector) == 384


class TestUpdatePayload:
    def test_update_payload_by_string_id(self):
        from qql.ast_nodes import UpdatePayloadStmt
        node = parse("UPDATE articles SET PAYLOAD WHERE id = 'abc-123' {'year': 2025}")
        assert isinstance(node, UpdatePayloadStmt)
        assert node.collection == "articles"
        assert node.point_id == "abc-123"
        assert node.payload == {"year": 2025}
        assert node.query_filter is None

    def test_update_payload_by_integer_id(self):
        from qql.ast_nodes import UpdatePayloadStmt
        node = parse("UPDATE articles SET PAYLOAD WHERE id = 42 {'status': 'active'}")
        assert isinstance(node, UpdatePayloadStmt)
        assert node.point_id == 42
        assert node.payload == {"status": "active"}

    def test_update_payload_by_filter(self):
        from qql.ast_nodes import UpdatePayloadStmt
        node = parse(
            "UPDATE articles SET PAYLOAD WHERE category = 'draft' {'status': 'published'}"
        )
        assert isinstance(node, UpdatePayloadStmt)
        assert node.point_id is None
        assert node.query_filter is not None
        assert node.payload == {"status": "published"}

    def test_update_payload_compound_filter(self):
        from qql.ast_nodes import UpdatePayloadStmt, AndExpr
        node = parse(
            "UPDATE articles SET PAYLOAD WHERE year < 2020 AND status = 'draft' "
            "{'archived': true}"
        )
        assert isinstance(node, UpdatePayloadStmt)
        assert isinstance(node.query_filter, AndExpr)
        assert node.payload == {"archived": True}

    def test_update_payload_dict_values_preserved(self):
        from qql.ast_nodes import UpdatePayloadStmt
        node = parse(
            "UPDATE articles SET PAYLOAD WHERE id = 1 "
            "{'title': 'New Title', 'year': 2025, 'score': 0.99}"
        )
        assert isinstance(node, UpdatePayloadStmt)
        assert node.payload["title"] == "New Title"
        assert node.payload["year"] == 2025
        assert node.payload["score"] == pytest.approx(0.99)

    def test_update_payload_collection_stored(self):
        node = parse("UPDATE my_notes SET PAYLOAD WHERE id = 7 {'tag': 'ai'}")
        assert node.collection == "my_notes"

    def test_update_payload_missing_dict_raises(self):
        with pytest.raises(QQLSyntaxError):
            parse("UPDATE articles SET PAYLOAD WHERE id = 1")

    def test_update_payload_dotted_filter_field(self):
        from qql.ast_nodes import UpdatePayloadStmt
        node = parse(
            "UPDATE articles SET PAYLOAD WHERE meta.author = 'alice' {'reviewed': true}"
        )
        assert isinstance(node, UpdatePayloadStmt)
        assert node.query_filter is not None
        assert node.payload == {"reviewed": True}


# ── PR #28 review gap fixes ───────────────────────────────────────────────────

class TestSearchGroupByValidation:
    """Parser-level validation added for PR #28 gaps 2 and 2."""

    def test_group_size_zero_raises(self):
        with pytest.raises(QQLSyntaxError, match="GROUP_SIZE must be a positive integer"):
            parse("SEARCH articles SIMILAR TO 'q' LIMIT 5 GROUP BY category GROUP_SIZE 0")

    def test_group_size_negative_raises(self):
        with pytest.raises(QQLSyntaxError, match="GROUP_SIZE must be a positive integer"):
            parse("SEARCH articles SIMILAR TO 'q' LIMIT 5 GROUP BY category GROUP_SIZE -1")


class TestUpdateVectorValidation:
    """PR #28 gap 11 — non-numeric vector elements should raise QQLSyntaxError."""

    def test_non_numeric_string_element_raises(self):
        with pytest.raises(QQLSyntaxError, match="Vector elements must be numeric"):
            parse("UPDATE articles SET VECTOR WHERE id = 1 ['abc', 0.2, 0.3]")

    def test_none_element_raises(self):
        # null parsed as Python None → TypeError → QQLSyntaxError
        with pytest.raises(QQLSyntaxError, match="Vector elements must be numeric"):
            parse("UPDATE articles SET VECTOR WHERE id = 1 [null, 0.2]")

    def test_boolean_true_element_raises(self):
        # bool is a subclass of int — float(True) == 1.0 would silently pass
        # without an explicit isinstance(v, bool) guard.
        with pytest.raises(QQLSyntaxError, match="boolean values are not allowed"):
            parse("UPDATE articles SET VECTOR WHERE id = 1 [true, 0.2, 0.3]")

    def test_boolean_false_element_raises(self):
        with pytest.raises(QQLSyntaxError, match="boolean values are not allowed"):
            parse("UPDATE articles SET VECTOR WHERE id = 1 [false, 0.5]")


class TestUpdateSetInvalidTargetMessage:
    """PR #28 gap 16 — explicit error message for bad SET target."""

    def test_invalid_set_target_message(self):
        with pytest.raises(QQLSyntaxError, match="Expected VECTOR or PAYLOAD after SET"):
            parse("UPDATE articles SET FOOBAR WHERE id = 1 [0.1]")
