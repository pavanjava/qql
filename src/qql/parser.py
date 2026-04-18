from typing import Any

from .ast_nodes import (
    ASTNode,
    AndExpr,
    BetweenExpr,
    CompareExpr,
    CreateCollectionStmt,
    DeleteStmt,
    DropCollectionStmt,
    FilterExpr,
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
from .exceptions import QQLSyntaxError
from .lexer import Token, TokenKind

# Comparison operator token → string symbol mapping
_CMP_OPS: dict[TokenKind, str] = {
    TokenKind.EQUALS:     "=",
    TokenKind.NOT_EQUALS: "!=",
    TokenKind.GT:         ">",
    TokenKind.GTE:        ">=",
    TokenKind.LT:         "<",
    TokenKind.LTE:        "<=",
}


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
        hybrid: bool = False
        sparse_model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            if self._peek().kind == TokenKind.HYBRID:
                self._advance()  # consume HYBRID
                hybrid = True
                # Optional DENSE MODEL and/or SPARSE MODEL sub-clauses, any order
                while self._peek().kind in (TokenKind.DENSE, TokenKind.SPARSE):
                    sub = self._advance()
                    self._expect(TokenKind.MODEL)
                    m = self._expect(TokenKind.STRING).value
                    if sub.kind == TokenKind.DENSE:
                        model = m
                    else:
                        sparse_model = m
            else:
                self._expect(TokenKind.MODEL)
                model = self._expect(TokenKind.STRING).value
        return InsertStmt(
            collection=collection, values=values, model=model,
            hybrid=hybrid, sparse_model=sparse_model,
        )

    def _parse_create(self) -> CreateCollectionStmt:
        self._expect(TokenKind.CREATE)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        hybrid: bool = False
        model: str | None = None

        if self._peek().kind == TokenKind.HYBRID:
            # Bare HYBRID shorthand — backward compat
            self._advance()
            hybrid = True
        elif self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            if self._peek().kind == TokenKind.HYBRID:
                self._advance()  # consume HYBRID
                hybrid = True
                # Optional DENSE MODEL sub-clause
                if self._peek().kind == TokenKind.DENSE:
                    self._advance()  # consume DENSE
                    self._expect(TokenKind.MODEL)
                    model = self._expect(TokenKind.STRING).value
            else:
                self._expect(TokenKind.MODEL)
                model = self._expect(TokenKind.STRING).value

        return CreateCollectionStmt(
            collection=collection,
            hybrid=hybrid,
            model=model,
        )

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

        with_clause: SearchWith | None = None
        if self._peek().kind == TokenKind.EXACT:
            self._advance()
            with_clause = SearchWith(exact=True)

        model: str | None = None
        hybrid: bool = False
        sparse_only: bool = False
        sparse_model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            if self._peek().kind == TokenKind.HYBRID:
                self._advance()  # consume HYBRID
                hybrid = True
                # Optional DENSE MODEL and/or SPARSE MODEL sub-clauses, any order
                while self._peek().kind in (TokenKind.DENSE, TokenKind.SPARSE):
                    sub = self._advance()
                    self._expect(TokenKind.MODEL)
                    m = self._expect(TokenKind.STRING).value
                    if sub.kind == TokenKind.DENSE:
                        model = m
                    else:
                        sparse_model = m
            elif self._peek().kind == TokenKind.SPARSE:
                self._advance()  # consume SPARSE
                sparse_only = True
                if self._peek().kind == TokenKind.MODEL:
                    self._advance()  # consume MODEL
                    sparse_model = self._expect(TokenKind.STRING).value
            else:
                self._expect(TokenKind.MODEL)
                model = self._expect(TokenKind.STRING).value
        query_filter: FilterExpr | None = None
        if self._peek().kind == TokenKind.WHERE:
            self._advance()  # consume WHERE
            query_filter = self._parse_filter_expr()
        rerank: bool = False
        rerank_model: str | None = None
        if self._peek().kind == TokenKind.RERANK:
            self._advance()  # consume RERANK
            rerank = True
            if self._peek().kind == TokenKind.MODEL:
                self._advance()  # consume MODEL
                rerank_model = self._expect(TokenKind.STRING).value
        if self._peek().kind == TokenKind.EXACT:
            self._advance()
            if with_clause is None:
                with_clause = SearchWith(exact=True)
            else:
                with_clause = SearchWith(
                    hnsw_ef=with_clause.hnsw_ef,
                    exact=True,
                    acorn=with_clause.acorn,
                )
        if self._peek().kind == TokenKind.WITH:
            self._advance()  # consume WITH
            parsed_with = self._parse_with_clause()
            if with_clause is None:
                with_clause = parsed_with
            else:
                with_clause = SearchWith(
                    hnsw_ef=parsed_with.hnsw_ef or with_clause.hnsw_ef,
                    exact=parsed_with.exact or with_clause.exact,
                    acorn=parsed_with.acorn or with_clause.acorn,
                )
        return SearchStmt(
            collection=collection,
            query_text=query_text,
            limit=limit,
            model=model,
            hybrid=hybrid,
            sparse_only=sparse_only,
            sparse_model=sparse_model,
            query_filter=query_filter,
            rerank=rerank,
            rerank_model=rerank_model,
            with_clause=with_clause,
        )

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

    # ── WHERE clause filter parsing (precedence: NOT > AND > OR) ─────────

    def _parse_filter_expr(self) -> FilterExpr:
        """filter_or ::= filter_and { OR filter_and }"""
        left = self._parse_filter_and()
        if self._peek().kind != TokenKind.OR:
            return left
        operands: list[FilterExpr] = [left]
        while self._peek().kind == TokenKind.OR:
            self._advance()  # consume OR
            operands.append(self._parse_filter_and())
        return OrExpr(operands=tuple(operands))

    def _parse_filter_and(self) -> FilterExpr:
        """filter_and ::= filter_not { AND filter_not }"""
        left = self._parse_filter_not()
        if self._peek().kind != TokenKind.AND:
            return left
        operands: list[FilterExpr] = [left]
        while self._peek().kind == TokenKind.AND:
            self._advance()  # consume AND
            operands.append(self._parse_filter_not())
        return AndExpr(operands=tuple(operands))

    def _parse_filter_not(self) -> FilterExpr:
        """filter_not ::= NOT filter_not | filter_primary"""
        if self._peek().kind == TokenKind.NOT:
            self._advance()  # consume NOT
            return NotExpr(operand=self._parse_filter_not())  # right-recursive
        return self._parse_filter_primary()

    def _parse_filter_primary(self) -> FilterExpr:
        """filter_primary ::= '(' filter_expr ')' | predicate"""
        if self._peek().kind == TokenKind.LPAREN:
            self._advance()  # consume (
            expr = self._parse_filter_expr()
            self._expect(TokenKind.RPAREN)
            return expr
        return self._parse_predicate()

    def _parse_predicate(self) -> FilterExpr:
        """All leaf filter conditions."""
        field = self._parse_field_path()
        tok = self._peek()

        # ── IS NULL / IS NOT NULL / IS EMPTY / IS NOT EMPTY ──────────────
        if tok.kind == TokenKind.IS:
            self._advance()  # consume IS
            if self._peek().kind == TokenKind.NOT:
                self._advance()  # consume NOT
                if self._peek().kind == TokenKind.NULL:
                    self._advance()
                    return IsNotNullExpr(field=field)
                if self._peek().kind == TokenKind.EMPTY:
                    self._advance()
                    return IsNotEmptyExpr(field=field)
                raise QQLSyntaxError(
                    "Expected NULL or EMPTY after IS NOT", self._peek().pos
                )
            if self._peek().kind == TokenKind.NULL:
                self._advance()
                return IsNullExpr(field=field)
            if self._peek().kind == TokenKind.EMPTY:
                self._advance()
                return IsEmptyExpr(field=field)
            raise QQLSyntaxError(
                "Expected NULL, NOT NULL, EMPTY, or NOT EMPTY after IS", self._peek().pos
            )

        # ── IN ( ... ) ────────────────────────────────────────────────────
        if tok.kind == TokenKind.IN:
            self._advance()  # consume IN
            values = self._parse_literal_list()
            return InExpr(field=field, values=tuple(values))

        # ── NOT IN ( ... ) ────────────────────────────────────────────────
        if tok.kind == TokenKind.NOT:
            self._advance()  # consume NOT
            self._expect(TokenKind.IN)
            values = self._parse_literal_list()
            return NotInExpr(field=field, values=tuple(values))

        # ── BETWEEN low AND high ──────────────────────────────────────────
        if tok.kind == TokenKind.BETWEEN:
            self._advance()  # consume BETWEEN
            low = self._parse_number()
            self._expect(TokenKind.AND)  # consumes AND as separator (not logical AND)
            high = self._parse_number()
            return BetweenExpr(field=field, low=low, high=high)

        # ── MATCH / MATCH ANY / MATCH PHRASE ─────────────────────────────
        if tok.kind == TokenKind.MATCH:
            self._advance()  # consume MATCH
            if self._peek().kind == TokenKind.ANY:
                self._advance()
                text = self._expect(TokenKind.STRING).value
                return MatchAnyExpr(field=field, text=text)
            if self._peek().kind == TokenKind.PHRASE:
                self._advance()
                text = self._expect(TokenKind.STRING).value
                return MatchPhraseExpr(field=field, text=text)
            # plain MATCH — all terms required
            text = self._expect(TokenKind.STRING).value
            return MatchTextExpr(field=field, text=text)

        # ── Comparison operators: =, !=, >, >=, <, <= ────────────────────
        if tok.kind in _CMP_OPS:
            op = _CMP_OPS[tok.kind]
            self._advance()
            value = self._parse_literal()
            return CompareExpr(field=field, op=op, value=value)

        raise QQLSyntaxError(
            f"Expected a filter operator after field '{field}', got '{tok.value}'",
            tok.pos,
        )

    # ── Filter parsing helpers ────────────────────────────────────────────

    def _parse_field_path(self) -> str:
        """Dot-notation paths are already single IDENTIFIER tokens from the lexer."""
        tok = self._peek()
        if tok.kind != TokenKind.IDENTIFIER:
            raise QQLSyntaxError(
                f"Expected a field name, got '{tok.value}'", tok.pos
            )
        self._advance()
        return tok.value

    def _parse_literal(self) -> str | int | float:
        """STRING | INTEGER | FLOAT"""
        tok = self._peek()
        if tok.kind == TokenKind.STRING:
            self._advance()
            return tok.value
        if tok.kind == TokenKind.INTEGER:
            self._advance()
            return int(tok.value)
        if tok.kind == TokenKind.FLOAT:
            self._advance()
            return float(tok.value)
        raise QQLSyntaxError(
            f"Expected a literal value (string, integer, or float), got '{tok.value}'",
            tok.pos,
        )

    def _parse_number(self) -> int | float:
        """INTEGER | FLOAT only (used by BETWEEN)."""
        tok = self._peek()
        if tok.kind == TokenKind.INTEGER:
            self._advance()
            return int(tok.value)
        if tok.kind == TokenKind.FLOAT:
            self._advance()
            return float(tok.value)
        raise QQLSyntaxError(
            f"Expected a number, got '{tok.value}'", tok.pos
        )

    def _parse_literal_list(self) -> list[str | int | float]:
        """'(' literal { ',' literal } [','] ')'  — used by IN / NOT IN."""
        self._expect(TokenKind.LPAREN)
        items: list[str | int | float] = []
        if self._peek().kind == TokenKind.RPAREN:
            self._advance()
            return items
        while True:
            items.append(self._parse_literal())
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                if self._peek().kind == TokenKind.RPAREN:
                    break  # trailing comma allowed
            else:
                break
        self._expect(TokenKind.RPAREN)
        return items

    # ── Dict / value parsers (for INSERT VALUES) ──────────────────────────

    def _parse_identifier(self) -> str:
        """Accept either a bare IDENTIFIER or a quoted STRING as a collection name."""
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
                if self._peek().kind == TokenKind.RBRACE:
                    break  # trailing comma
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
        if tok.kind == TokenKind.NULL:
            # NULL is now a keyword token
            self._advance()
            return None
        if tok.kind == TokenKind.IDENTIFIER:
            upper = tok.value.upper()
            if upper == "TRUE":
                self._advance()
                return True
            if upper == "FALSE":
                self._advance()
                return False
            if upper == "NULL":
                # Fallback: handle 'null' that arrived as IDENTIFIER (shouldn't happen
                # after lexer change, but kept for safety)
                self._advance()
                return None
            self._advance()
            return tok.value
        if tok.kind == TokenKind.LBRACE:
            return self._parse_dict()
        if tok.kind == TokenKind.LBRACKET:
            return self._parse_list()
        raise QQLSyntaxError(f"Unexpected value token '{tok.value}'", tok.pos)

    # ── WITH clause: { hnsw_ef: N, exact: true, acorn: true } ──

    def _parse_with_clause(self) -> SearchWith:
        self._expect(TokenKind.LBRACE)
        hnsw_ef: int | None = None
        exact: bool = False
        acorn: bool = False
        while self._peek().kind != TokenKind.RBRACE:
            key_tok = self._peek()
            if key_tok.kind not in (
                TokenKind.IDENTIFIER,
                TokenKind.EXACT,
                TokenKind.ACORN,
            ):
                raise QQLSyntaxError(
                    f"Expected a WITH parameter name, got '{key_tok.value}'",
                    key_tok.pos,
                )
            self._advance()
            key = key_tok.value.lower()
            self._expect(TokenKind.COLON)
            if key == "hnsw_ef":
                hnsw_ef = int(self._expect(TokenKind.INTEGER).value)
            elif key == "exact":
                exact = self._parse_bool()
            elif key == "acorn":
                acorn = self._parse_bool()
            else:
                raise QQLSyntaxError(
                    f"Unknown WITH parameter '{key}'. Expected: hnsw_ef, exact, acorn",
                    key_tok.pos,
                )
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                if self._peek().kind == TokenKind.RBRACE:
                    break
            else:
                break
        self._expect(TokenKind.RBRACE)
        return SearchWith(
            hnsw_ef=hnsw_ef,
            exact=exact,
            acorn=acorn,
        )

    def _parse_bool(self) -> bool:
        tok = self._peek()
        if tok.kind == TokenKind.IDENTIFIER:
            val = tok.value.upper()
            self._advance()
            if val == "TRUE":
                return True
            if val == "FALSE":
                return False
        raise QQLSyntaxError(f"Expected true or false, got '{tok.value}'", tok.pos)

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
