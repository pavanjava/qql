from enum import Enum, auto
from typing import NamedTuple

from .exceptions import QQLSyntaxError


class TokenKind(Enum):
    # Keywords
    INSERT = auto()
    INTO = auto()
    COLLECTION = auto()
    VALUES = auto()
    USING = auto()
    MODEL = auto()
    CREATE = auto()
    DROP = auto()
    SHOW = auto()
    COLLECTIONS = auto()
    SEARCH = auto()
    SIMILAR = auto()
    TO = auto()
    LIMIT = auto()
    DELETE = auto()
    FROM = auto()
    WHERE = auto()
    ID = auto()
    # Literals & names
    IDENTIFIER = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    # Punctuation
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COLON = auto()
    COMMA = auto()
    EQUALS = auto()
    # Control
    EOF = auto()


_KEYWORDS: dict[str, TokenKind] = {
    "INSERT": TokenKind.INSERT,
    "INTO": TokenKind.INTO,
    "COLLECTION": TokenKind.COLLECTION,
    "VALUES": TokenKind.VALUES,
    "USING": TokenKind.USING,
    "MODEL": TokenKind.MODEL,
    "CREATE": TokenKind.CREATE,
    "DROP": TokenKind.DROP,
    "SHOW": TokenKind.SHOW,
    "COLLECTIONS": TokenKind.COLLECTIONS,
    "SEARCH": TokenKind.SEARCH,
    "SIMILAR": TokenKind.SIMILAR,
    "TO": TokenKind.TO,
    "LIMIT": TokenKind.LIMIT,
    "DELETE": TokenKind.DELETE,
    "FROM": TokenKind.FROM,
    "WHERE": TokenKind.WHERE,
    "ID": TokenKind.ID,
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

            # Single-character punctuation
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
            elif ch == ":":
                tokens.append(Token(TokenKind.COLON, ":", i))
                i += 1
            elif ch == ",":
                tokens.append(Token(TokenKind.COMMA, ",", i))
                i += 1
            elif ch == "=":
                tokens.append(Token(TokenKind.EQUALS, "=", i))
                i += 1

            # String literals
            elif ch in ('"', "'"):
                start = i
                quote = ch
                i += 1
                buf: list[str] = []
                while i < n:
                    if query[i] == "\\" and i + 1 < n:
                        # Handle escape sequences
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

            # Numbers: optional leading minus
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

            # Identifiers and keywords
            elif ch.isalpha() or ch == "_":
                start = i
                while i < n and (query[i].isalnum() or query[i] == "_"):
                    i += 1
                word = query[start:i]
                upper = word.upper()
                kind = _KEYWORDS.get(upper, TokenKind.IDENTIFIER)
                tokens.append(Token(kind, word, start))

            else:
                raise QQLSyntaxError(f"Unexpected character '{ch}'", i)

        tokens.append(Token(TokenKind.EOF, "", n))
        return tokens
