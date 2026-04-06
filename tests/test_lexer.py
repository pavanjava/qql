import pytest

from qql.exceptions import QQLSyntaxError
from qql.lexer import Lexer, TokenKind


def tokenize(q):
    return Lexer().tokenize(q)


def kinds(q):
    return [t.kind for t in tokenize(q)]


class TestKeywords:
    def test_insert_keywords(self):
        ks = kinds("INSERT INTO COLLECTION foo VALUES")
        assert ks[:5] == [
            TokenKind.INSERT,
            TokenKind.INTO,
            TokenKind.COLLECTION,
            TokenKind.IDENTIFIER,
            TokenKind.VALUES,
        ]

    def test_keywords_case_insensitive(self):
        ks = kinds("insert into collection foo values")
        assert ks[0] == TokenKind.INSERT
        assert ks[1] == TokenKind.INTO

    def test_show_collections(self):
        ks = kinds("SHOW COLLECTIONS")
        assert ks[:2] == [TokenKind.SHOW, TokenKind.COLLECTIONS]

    def test_search_keywords(self):
        ks = kinds("SEARCH mycol SIMILAR TO 'hi' LIMIT 5")
        assert ks[0] == TokenKind.SEARCH
        assert ks[2] == TokenKind.SIMILAR
        assert ks[3] == TokenKind.TO
        assert ks[5] == TokenKind.LIMIT

    def test_delete_keywords(self):
        ks = kinds("DELETE FROM foo WHERE id = 'abc'")
        assert ks[:4] == [TokenKind.DELETE, TokenKind.FROM, TokenKind.IDENTIFIER, TokenKind.WHERE]


class TestLiterals:
    def test_double_quoted_string(self):
        tokens = tokenize('"hello world"')
        assert tokens[0].kind == TokenKind.STRING
        assert tokens[0].value == "hello world"

    def test_single_quoted_string(self):
        tokens = tokenize("'hello'")
        assert tokens[0].kind == TokenKind.STRING
        assert tokens[0].value == "hello"

    def test_integer(self):
        tokens = tokenize("42")
        assert tokens[0].kind == TokenKind.INTEGER
        assert tokens[0].value == "42"

    def test_negative_integer(self):
        tokens = tokenize("-7")
        assert tokens[0].kind == TokenKind.INTEGER
        assert tokens[0].value == "-7"

    def test_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].kind == TokenKind.FLOAT
        assert tokens[0].value == "3.14"

    def test_identifier(self):
        tokens = tokenize("my_collection_1")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "my_collection_1"


class TestPunctuation:
    def test_braces_colons_commas(self):
        ks = kinds("{ 'a' : 1 , 'b' : 2 }")
        assert TokenKind.LBRACE in ks
        assert TokenKind.RBRACE in ks
        assert TokenKind.COLON in ks
        assert TokenKind.COMMA in ks

    def test_brackets(self):
        ks = kinds("[ 1, 2 ]")
        assert ks[0] == TokenKind.LBRACKET
        assert ks[-2] == TokenKind.RBRACKET


class TestErrors:
    def test_unterminated_string(self):
        with pytest.raises(QQLSyntaxError, match="Unterminated"):
            tokenize('"not closed')

    def test_unexpected_character(self):
        with pytest.raises(QQLSyntaxError, match="Unexpected character"):
            tokenize("@bad")

    def test_error_includes_position(self):
        with pytest.raises(QQLSyntaxError) as exc_info:
            tokenize("abc @")
        assert exc_info.value.pos is not None


class TestEOF:
    def test_ends_with_eof(self):
        tokens = tokenize("hello")
        assert tokens[-1].kind == TokenKind.EOF
