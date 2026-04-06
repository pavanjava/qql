from typing import Any

from .ast_nodes import (
    ASTNode,
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    InsertStmt,
    SearchStmt,
    ShowCollectionsStmt,
)
from .exceptions import QQLSyntaxError
from .lexer import Token, TokenKind


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ── Public entry point ────────────────────────────────────────────────

    def parse(self) -> ASTNode:
        tok = self._peek()
        if tok.kind == TokenKind.INSERT:
            node = self._parse_insert()
        elif tok.kind == TokenKind.CREATE:
            node = self._parse_create()
        elif tok.kind == TokenKind.DROP:
            node = self._parse_drop()
        elif tok.kind == TokenKind.SHOW:
            node = self._parse_show()
        elif tok.kind == TokenKind.SEARCH:
            node = self._parse_search()
        elif tok.kind == TokenKind.DELETE:
            node = self._parse_delete()
        else:
            raise QQLSyntaxError(
                f"Unexpected token '{tok.value}'; expected a QQL statement keyword",
                tok.pos,
            )
        self._expect(TokenKind.EOF)
        return node

    # ── Statement parsers ─────────────────────────────────────────────────

    def _parse_insert(self) -> InsertStmt:
        self._expect(TokenKind.INSERT)
        self._expect(TokenKind.INTO)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        self._expect(TokenKind.VALUES)
        values = self._parse_dict()
        model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            self._expect(TokenKind.MODEL)
            model = self._expect(TokenKind.STRING).value
        return InsertStmt(collection=collection, values=values, model=model)

    def _parse_create(self) -> CreateCollectionStmt:
        self._expect(TokenKind.CREATE)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        return CreateCollectionStmt(collection=collection)

    def _parse_drop(self) -> DropCollectionStmt:
        self._expect(TokenKind.DROP)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        return DropCollectionStmt(collection=collection)

    def _parse_show(self) -> ShowCollectionsStmt:
        self._expect(TokenKind.SHOW)
        self._expect(TokenKind.COLLECTIONS)
        return ShowCollectionsStmt()

    def _parse_search(self) -> SearchStmt:
        self._expect(TokenKind.SEARCH)
        collection = self._parse_identifier()
        self._expect(TokenKind.SIMILAR)
        self._expect(TokenKind.TO)
        query_text = self._expect(TokenKind.STRING).value
        self._expect(TokenKind.LIMIT)
        limit = int(self._expect(TokenKind.INTEGER).value)
        model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()
            self._expect(TokenKind.MODEL)
            model = self._expect(TokenKind.STRING).value
        return SearchStmt(collection=collection, query_text=query_text, limit=limit, model=model)

    def _parse_delete(self) -> DeleteStmt:
        self._expect(TokenKind.DELETE)
        self._expect(TokenKind.FROM)
        collection = self._parse_identifier()
        self._expect(TokenKind.WHERE)
        self._expect(TokenKind.ID)
        self._expect(TokenKind.EQUALS)
        tok = self._peek()
        if tok.kind == TokenKind.STRING:
            self._advance()
            point_id: str | int = tok.value
        elif tok.kind == TokenKind.INTEGER:
            self._advance()
            point_id = int(tok.value)
        else:
            raise QQLSyntaxError(
                f"Expected string or integer for point id, got '{tok.value}'", tok.pos
            )
        return DeleteStmt(collection=collection, point_id=point_id)

    # ── Value parsers ─────────────────────────────────────────────────────

    def _parse_identifier(self) -> str:
        """Accept either a bare IDENTIFIER or a quoted STRING as a name."""
        tok = self._peek()
        if tok.kind == TokenKind.IDENTIFIER:
            self._advance()
            return tok.value
        if tok.kind == TokenKind.STRING:
            self._advance()
            return tok.value
        raise QQLSyntaxError(
            f"Expected identifier or quoted name, got '{tok.value}'", tok.pos
        )

    def _parse_dict(self) -> dict[str, Any]:
        self._expect(TokenKind.LBRACE)
        result: dict[str, Any] = {}
        if self._peek().kind == TokenKind.RBRACE:
            self._advance()
            return result
        while True:
            key_tok = self._peek()
            if key_tok.kind not in (TokenKind.STRING, TokenKind.IDENTIFIER):
                raise QQLSyntaxError(
                    f"Expected string key in dict, got '{key_tok.value}'", key_tok.pos
                )
            self._advance()
            key = key_tok.value
            self._expect(TokenKind.COLON)
            value = self._parse_value()
            result[key] = value
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                # Allow trailing comma
                if self._peek().kind == TokenKind.RBRACE:
                    break
            else:
                break
        self._expect(TokenKind.RBRACE)
        return result

    def _parse_list(self) -> list[Any]:
        self._expect(TokenKind.LBRACKET)
        items: list[Any] = []
        if self._peek().kind == TokenKind.RBRACKET:
            self._advance()
            return items
        while True:
            items.append(self._parse_value())
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                if self._peek().kind == TokenKind.RBRACKET:
                    break
            else:
                break
        self._expect(TokenKind.RBRACKET)
        return items

    def _parse_value(self) -> Any:
        tok = self._peek()
        if tok.kind == TokenKind.STRING:
            self._advance()
            return tok.value
        if tok.kind == TokenKind.FLOAT:
            self._advance()
            return float(tok.value)
        if tok.kind == TokenKind.INTEGER:
            self._advance()
            return int(tok.value)
        if tok.kind == TokenKind.IDENTIFIER:
            upper = tok.value.upper()
            if upper == "TRUE":
                self._advance()
                return True
            if upper == "FALSE":
                self._advance()
                return False
            if upper == "NULL":
                self._advance()
                return None
            self._advance()
            return tok.value
        if tok.kind == TokenKind.LBRACE:
            return self._parse_dict()
        if tok.kind == TokenKind.LBRACKET:
            return self._parse_list()
        raise QQLSyntaxError(f"Unexpected value token '{tok.value}'", tok.pos)

    # ── Token stream helpers ──────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind != TokenKind.EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, value: str | None = None) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise QQLSyntaxError(
                f"Expected {kind.name}"
                + (f" '{value}'" if value else "")
                + f", got '{tok.value}'",
                tok.pos,
            )
        if value is not None and tok.value.upper() != value.upper():
            raise QQLSyntaxError(
                f"Expected '{value}', got '{tok.value}'", tok.pos
            )
        return self._advance()
