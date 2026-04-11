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


class TestNewOperators:
    def test_not_equals(self):
        tokens = tokenize("field != 'x'")
        assert tokens[1].kind == TokenKind.NOT_EQUALS
        assert tokens[1].value == "!="

    def test_gt(self):
        tokens = tokenize("score > 0.5")
        assert tokens[1].kind == TokenKind.GT
        assert tokens[1].value == ">"

    def test_gte(self):
        tokens = tokenize("score >= 0.5")
        assert tokens[1].kind == TokenKind.GTE
        assert tokens[1].value == ">="

    def test_lt(self):
        tokens = tokenize("year < 2024")
        assert tokens[1].kind == TokenKind.LT
        assert tokens[1].value == "<"

    def test_lte(self):
        tokens = tokenize("year <= 2023")
        assert tokens[1].kind == TokenKind.LTE
        assert tokens[1].value == "<="

    def test_lparen_rparen(self):
        ks = kinds("(a OR b)")
        assert TokenKind.LPAREN in ks
        assert TokenKind.RPAREN in ks

    def test_filter_keywords(self):
        ks = kinds("AND OR NOT IN BETWEEN IS NULL EMPTY MATCH ANY PHRASE")
        assert TokenKind.AND     in ks
        assert TokenKind.OR      in ks
        assert TokenKind.NOT     in ks
        assert TokenKind.IN      in ks
        assert TokenKind.BETWEEN in ks
        assert TokenKind.IS      in ks
        assert TokenKind.NULL    in ks
        assert TokenKind.EMPTY   in ks
        assert TokenKind.MATCH   in ks
        assert TokenKind.ANY     in ks
        assert TokenKind.PHRASE  in ks

    def test_filter_keywords_case_insensitive(self):
        ks = kinds("and or not in between is null empty match any phrase")
        assert TokenKind.AND in ks
        assert TokenKind.OR  in ks
        assert TokenKind.NOT in ks

    def test_dotted_identifier(self):
        tokens = tokenize("meta.source")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "meta.source"

    def test_three_level_dotted_identifier(self):
        tokens = tokenize("a.b.c")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "a.b.c"

    def test_nested_array_path(self):
        tokens = tokenize("country.cities[].population")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "country.cities[].population"

    def test_gt_does_not_consume_equals_sign(self):
        # ">" followed by non-"=" should be GT only
        tokens = tokenize("a > b")
        assert tokens[1].kind == TokenKind.GT

    def test_bare_exclamation_raises(self):
        with pytest.raises(QQLSyntaxError):
            tokenize("field ! 'x'")


class TestEOF:
    def test_ends_with_eof(self):
        tokens = tokenize("hello")
        assert tokens[-1].kind == TokenKind.EOF


class TestHybridKeyword:
    def test_hybrid_keyword_uppercase(self):
        ks = kinds("HYBRID")
        assert ks[0] == TokenKind.HYBRID

    def test_hybrid_keyword_lowercase(self):
        ks = kinds("hybrid")
        assert ks[0] == TokenKind.HYBRID

    def test_dense_keyword(self):
        ks = kinds("DENSE")
        assert ks[0] == TokenKind.DENSE

    def test_dense_keyword_lowercase(self):
        ks = kinds("dense")
        assert ks[0] == TokenKind.DENSE

    def test_sparse_keyword(self):
        ks = kinds("SPARSE")
        assert ks[0] == TokenKind.SPARSE

    def test_sparse_keyword_lowercase(self):
        ks = kinds("sparse")
        assert ks[0] == TokenKind.SPARSE

    def test_hybrid_in_create_statement(self):
        ks = kinds("CREATE COLLECTION articles HYBRID")
        assert ks[3] == TokenKind.HYBRID

    def test_hybrid_in_search_statement(self):
        ks = kinds("SEARCH col SIMILAR TO 'q' LIMIT 5 USING HYBRID")
        assert TokenKind.HYBRID in ks

    def test_dense_as_identifier_in_dotted_path(self):
        tokens = tokenize("dense.field")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "dense.field"

    def test_sparse_as_identifier_in_dotted_path(self):
        tokens = tokenize("sparse.value")
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "sparse.value"
