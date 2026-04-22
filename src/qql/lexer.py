from enum import Enum, auto
from typing import NamedTuple

from .exceptions import QQLSyntaxError


class TokenKind(Enum):
    # ── Statement keywords ────────────────────────────────────────────────
    INSERT = auto()
    BULK = auto()
    INTO = auto()
    COLLECTION = auto()
    VALUES = auto()
    USING = auto()
    MODEL = auto()
    HYBRID = auto()
    DENSE = auto()
    SPARSE = auto()
    RERANK = auto()
    EXACT = auto()
    WITH = auto()
    ACORN = auto()
    CREATE = auto()
    DROP = auto()
    SHOW = auto()
    COLLECTIONS = auto()
    SEARCH = auto()
    RECOMMEND = auto()
    POSITIVE = auto()
    NEGATIVE = auto()
    IDS = auto()
    STRATEGY = auto()
    SIMILAR = auto()
    TO = auto()
    LIMIT = auto()
    OFFSET = auto()
    SCORE = auto()
    THRESHOLD = auto()
    LOOKUP = auto()
    VECTOR = auto()
    DELETE = auto()
    FROM = auto()
    WHERE = auto()
    ID = auto()
    # ── Filter keywords ───────────────────────────────────────────────────
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    BETWEEN = auto()
    IS = auto()
    NULL = auto()
    EMPTY = auto()
    MATCH = auto()
    ANY = auto()
    PHRASE = auto()
    # ── Literals & names ─────────────────────────────────────────────────
    IDENTIFIER = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    # ── Punctuation ───────────────────────────────────────────────────────
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LPAREN = auto()
    RPAREN = auto()
    COLON = auto()
    COMMA = auto()
    EQUALS = auto()
    # ── Comparison operators ──────────────────────────────────────────────
    NOT_EQUALS = auto()  # !=
    GT = auto()  # >
    GTE = auto()  # >=
    LT = auto()  # <
    LTE = auto()  # <=
    # ── Control ───────────────────────────────────────────────────────────
    EOF = auto()


_KEYWORDS: dict[str, TokenKind] = {
    # Statement keywords
    "INSERT": TokenKind.INSERT,
    "BULK": TokenKind.BULK,
    "INTO": TokenKind.INTO,
    "COLLECTION": TokenKind.COLLECTION,
    "VALUES": TokenKind.VALUES,
    "USING": TokenKind.USING,
    "MODEL": TokenKind.MODEL,
    "HYBRID": TokenKind.HYBRID,
    "DENSE": TokenKind.DENSE,
    "SPARSE": TokenKind.SPARSE,
    "RERANK": TokenKind.RERANK,
    "EXACT": TokenKind.EXACT,
    "WITH": TokenKind.WITH,
    "ACORN": TokenKind.ACORN,
    "CREATE": TokenKind.CREATE,
    "DROP": TokenKind.DROP,
    "SHOW": TokenKind.SHOW,
    "COLLECTIONS": TokenKind.COLLECTIONS,
    "SEARCH": TokenKind.SEARCH,
    "RECOMMEND": TokenKind.RECOMMEND,
    "POSITIVE": TokenKind.POSITIVE,
    "NEGATIVE": TokenKind.NEGATIVE,
    "IDS": TokenKind.IDS,
    "STRATEGY": TokenKind.STRATEGY,
    "SIMILAR": TokenKind.SIMILAR,
    "TO": TokenKind.TO,
    "LIMIT": TokenKind.LIMIT,
    "OFFSET": TokenKind.OFFSET,
    "SCORE": TokenKind.SCORE,
    "THRESHOLD": TokenKind.THRESHOLD,
    "LOOKUP": TokenKind.LOOKUP,
    "VECTOR": TokenKind.VECTOR,
    "DELETE": TokenKind.DELETE,
    "FROM": TokenKind.FROM,
    "WHERE": TokenKind.WHERE,
    "ID": TokenKind.ID,
    # Filter keywords
    "AND": TokenKind.AND,
    "OR": TokenKind.OR,
    "NOT": TokenKind.NOT,
    "IN": TokenKind.IN,
    "BETWEEN": TokenKind.BETWEEN,
    "IS": TokenKind.IS,
    "NULL": TokenKind.NULL,
    "EMPTY": TokenKind.EMPTY,
    "MATCH": TokenKind.MATCH,
    "ANY": TokenKind.ANY,
    "PHRASE": TokenKind.PHRASE,
}


class Token(NamedTuple):
    kind: TokenKind
    value: str
    pos: int


class Lexer:
    def tokenize(self, query: str) -> list[Token]:
        tokens: list[Token] = []
        i = 0
        n = len(query)

        while i < n:
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue

            ch = query[i]

            # ── Braces / brackets / punctuation ──────────────────────────
            if ch == "{":
                tokens.append(Token(TokenKind.LBRACE, "{", i))
                i += 1
            elif ch == "}":
                tokens.append(Token(TokenKind.RBRACE, "}", i))
                i += 1
            elif ch == "[":
                tokens.append(Token(TokenKind.LBRACKET, "[", i))
                i += 1
            elif ch == "]":
                tokens.append(Token(TokenKind.RBRACKET, "]", i))
                i += 1
            elif ch == "(":
                tokens.append(Token(TokenKind.LPAREN, "(", i))
                i += 1
            elif ch == ")":
                tokens.append(Token(TokenKind.RPAREN, ")", i))
                i += 1
            elif ch == ":":
                tokens.append(Token(TokenKind.COLON, ":", i))
                i += 1
            elif ch == ",":
                tokens.append(Token(TokenKind.COMMA, ",", i))
                i += 1

            # ── Comparison operators (multi-char look-ahead) ──────────────
            elif ch == "=":
                tokens.append(Token(TokenKind.EQUALS, "=", i))
                i += 1
            elif ch == "!":
                if i + 1 < n and query[i + 1] == "=":
                    tokens.append(Token(TokenKind.NOT_EQUALS, "!=", i))
                    i += 2
                else:
                    raise QQLSyntaxError(f"Unexpected character '!'", i)
            elif ch == ">":
                if i + 1 < n and query[i + 1] == "=":
                    tokens.append(Token(TokenKind.GTE, ">=", i))
                    i += 2
                else:
                    tokens.append(Token(TokenKind.GT, ">", i))
                    i += 1
            elif ch == "<":
                if i + 1 < n and query[i + 1] == "=":
                    tokens.append(Token(TokenKind.LTE, "<=", i))
                    i += 2
                else:
                    tokens.append(Token(TokenKind.LT, "<", i))
                    i += 1

            # ── String literals ───────────────────────────────────────────
            elif ch in ('"', "'"):
                start = i
                quote = ch
                i += 1
                buf: list[str] = []
                while i < n:
                    if query[i] == "\\" and i + 1 < n:
                        next_ch = query[i + 1]
                        if next_ch == "n":
                            buf.append("\n")
                        elif next_ch == "t":
                            buf.append("\t")
                        elif next_ch in ('"', "'", "\\"):
                            buf.append(next_ch)
                        else:
                            buf.append("\\")
                            buf.append(next_ch)
                        i += 2
                    elif query[i] == quote:
                        i += 1
                        break
                    else:
                        buf.append(query[i])
                        i += 1
                else:
                    raise QQLSyntaxError("Unterminated string literal", start)
                tokens.append(Token(TokenKind.STRING, "".join(buf), start))

            # ── Numbers: optional leading minus ───────────────────────────
            elif ch.isdigit() or (ch == "-" and i + 1 < n and query[i + 1].isdigit()):
                start = i
                if ch == "-":
                    i += 1
                while i < n and query[i].isdigit():
                    i += 1
                if i < n and query[i] == "." and i + 1 < n and query[i + 1].isdigit():
                    i += 1  # consume "."
                    while i < n and query[i].isdigit():
                        i += 1
                    tokens.append(Token(TokenKind.FLOAT, query[start:i], start))
                else:
                    tokens.append(Token(TokenKind.INTEGER, query[start:i], start))

            # ── Identifiers, keywords, and dot-notation field paths ────────
            elif ch.isalpha() or ch == "_":
                start = i
                # Collect the base word
                while i < n and (query[i].isalnum() or query[i] == "_"):
                    i += 1
                # Extend for dotted field paths: consume ".word" and "[].word" segments
                # so that meta.source and country.cities[].population become single tokens.
                while i < n:
                    if query[i] == "." and i + 1 < n and (query[i + 1].isalpha() or query[i + 1] == "_"):
                        # ".identifier" segment
                        i += 1  # consume "."
                        while i < n and (query[i].isalnum() or query[i] == "_"):
                            i += 1
                    elif (
                        i + 2 < n
                        and query[i : i + 3] == "[]."
                        and i + 3 < n
                        and (query[i + 3].isalpha() or query[i + 3] == "_")
                    ):
                        # "[]." array marker segment
                        i += 3  # consume "[]."
                        while i < n and (query[i].isalnum() or query[i] == "_"):
                            i += 1
                    else:
                        break
                word = query[start:i]
                # Keyword lookup uses the uppercased first segment only for dotted paths
                # so that field names like "meta.from" are always IDENTIFIER, not keywords.
                first_segment = word.split(".")[0].upper()
                if "." not in word and first_segment in _KEYWORDS:
                    kind = _KEYWORDS[first_segment]
                else:
                    kind = TokenKind.IDENTIFIER
                tokens.append(Token(kind, word, start))

            else:
                raise QQLSyntaxError(f"Unexpected character '{ch}'", i)

        tokens.append(Token(TokenKind.EOF, "", n))
        return tokens
