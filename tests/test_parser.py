import pytest

from qql.ast_nodes import (
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertStmt,
    SearchStmt,
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
