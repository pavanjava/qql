from typing import Any

from .ast_nodes import (
    ASTNode,
    AndExpr,
    BetweenExpr,
    CompareExpr,
    CreateCollectionStmt,
    CreateIndexStmt,
    DeleteStmt,
    DropCollectionStmt,
    FilterExpr,
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
    OrExpr,
    QuantizationConfig,
    QuantizationType,
    QuantizationSearchWith,
    RecommendStmt,
    SelectStmt,
    ScrollStmt,
    SearchStmt,
    SearchWith,
    ShowCollectionStmt,
    ShowCollectionsStmt,
    UpdateVectorStmt,
    UpdatePayloadStmt,
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

_HYBRID_FUSION_VALUES = {"rrf", "dbsf"}


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
        elif tok.kind == TokenKind.SCROLL:
            node = self._parse_scroll()
        elif tok.kind == TokenKind.SELECT:
            node = self._parse_select()
        elif tok.kind == TokenKind.SEARCH:
            node = self._parse_search()
        elif tok.kind == TokenKind.RECOMMEND:
            node = self._parse_recommend()
        elif tok.kind == TokenKind.DELETE:
            node = self._parse_delete()
        elif tok.kind == TokenKind.UPDATE:
            node = self._parse_update()
        else:
            raise QQLSyntaxError(
                f"Unexpected token '{tok.value}'; expected a QQL statement keyword",
                tok.pos,
            )
        self._expect(TokenKind.EOF)
        return node

    # ── Statement parsers ─────────────────────────────────────────────────

    def _parse_insert(self) -> InsertStmt | InsertBulkStmt:
        self._expect(TokenKind.INSERT)
        if self._peek().kind == TokenKind.BULK:
            self._advance()  # consume BULK
            return self._parse_insert_bulk_body()
        # ── Standard single INSERT ────────────────────────────────────────
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

    def _parse_insert_bulk_body(self) -> InsertBulkStmt:
        self._expect(TokenKind.INTO)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        self._expect(TokenKind.VALUES)
        raw_list = self._parse_list()
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise QQLSyntaxError(
                    f"INSERT BULK VALUES item at index {i} must be a dict, "
                    f"got {type(item).__name__}",
                    0,
                )
        values_list: tuple[dict, ...] = tuple(raw_list)
        model: str | None = None
        hybrid: bool = False
        sparse_model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            if self._peek().kind == TokenKind.HYBRID:
                self._advance()  # consume HYBRID
                hybrid = True
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
        return InsertBulkStmt(
            collection=collection, values_list=values_list,
            model=model, hybrid=hybrid, sparse_model=sparse_model,
        )

    def _parse_create(self) -> CreateCollectionStmt | CreateIndexStmt:
        self._expect(TokenKind.CREATE)
        if self._peek().kind == TokenKind.COLLECTION:
            self._advance()
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

            # ── Optional QUANTIZE clause ──────────────────────────────────
            quantization: QuantizationConfig | None = None
            payload_m: int | None = None
            seen_quantize = False
            seen_hnsw = False
            while self._peek().kind in (TokenKind.QUANTIZE, TokenKind.HNSW):
                if self._peek().kind == TokenKind.QUANTIZE:
                    if seen_quantize:
                        raise QQLSyntaxError(
                            "QUANTIZE clause may only appear once",
                            self._peek().pos,
                        )
                    self._advance()  # consume QUANTIZE
                    quantization = self._parse_quantize_clause()
                    seen_quantize = True
                    continue

                if seen_hnsw:
                    raise QQLSyntaxError(
                        "HNSW clause may only appear once",
                        self._peek().pos,
                    )
                self._advance()  # consume HNSW
                payload_m = self._parse_collection_hnsw_clause()
                seen_hnsw = True

            return CreateCollectionStmt(
                collection=collection,
                hybrid=hybrid,
                model=model,
                quantization=quantization,
                payload_m=payload_m,
            )

        self._expect(TokenKind.INDEX)
        self._expect(TokenKind.ON)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        self._expect(TokenKind.FOR)
        field_name = self._parse_field_path()
        self._expect(TokenKind.TYPE)
        schema = self._expect(TokenKind.IDENTIFIER).value.lower()
        options: dict[str, Any] | None = None
        if self._peek().kind == TokenKind.WITH:
            self._advance()
            options = self._parse_dict()
        return CreateIndexStmt(
            collection=collection,
            field_name=field_name,
            schema=schema,
            options=options,
        )

    def _parse_collection_hnsw_clause(self) -> int:
        config = self._parse_dict()
        unknown_keys = set(config) - {"payload_m"}
        if unknown_keys:
            raise QQLSyntaxError(
                "Unknown HNSW parameter "
                f"'{sorted(unknown_keys)[0]}'. Expected: payload_m",
                0,
            )
        if "payload_m" not in config:
            raise QQLSyntaxError("HNSW clause requires payload_m", 0)
        payload_m = config["payload_m"]
        if not isinstance(payload_m, int) or isinstance(payload_m, bool) or payload_m <= 0:
            raise QQLSyntaxError("payload_m must be a positive integer", 0)
        return payload_m

    def _parse_quantize_clause(self) -> QuantizationConfig:
        """Parse: (SCALAR | BINARY | PRODUCT) [QUANTILE <float>] [ALWAYS RAM]

        Called immediately after the QUANTIZE token has been consumed.
        """
        tok = self._peek()

        if tok.kind == TokenKind.SCALAR:
            self._advance()
            quantile: float | None = None
            always_ram: bool = False
            if self._peek().kind == TokenKind.QUANTILE:
                self._advance()
                quantile_tok = self._peek()
                quantile = float(self._parse_number())
                if not 0.0 <= quantile <= 1.0:
                    raise QQLSyntaxError(
                        f"QUANTILE must be between 0 and 1 inclusive, got {quantile}",
                        quantile_tok.pos,
                    )
            if self._peek().kind == TokenKind.ALWAYS:
                self._advance()
                self._expect(TokenKind.RAM)
                always_ram = True
            return QuantizationConfig(
                type=QuantizationType.SCALAR,
                quantile=quantile,
                always_ram=always_ram,
            )

        if tok.kind == TokenKind.BINARY:
            self._advance()
            always_ram = False
            if self._peek().kind == TokenKind.ALWAYS:
                self._advance()
                self._expect(TokenKind.RAM)
                always_ram = True
            return QuantizationConfig(type=QuantizationType.BINARY, always_ram=always_ram)

        if tok.kind == TokenKind.PRODUCT:
            self._advance()
            always_ram = False
            if self._peek().kind == TokenKind.ALWAYS:
                self._advance()
                self._expect(TokenKind.RAM)
                always_ram = True
            return QuantizationConfig(type=QuantizationType.PRODUCT, always_ram=always_ram)

        if tok.kind == TokenKind.TURBO:
            self._advance()
            turbo_bits: float | None = None
            always_ram = False
            if self._peek().kind == TokenKind.BITS:
                self._advance()
                bits_tok = self._peek()
                raw = float(self._parse_number())
                if raw not in (1.0, 1.5, 2.0, 4.0):
                    raise QQLSyntaxError(
                        f"BITS must be one of 1, 1.5, 2, or 4 for TURBO quantization, got {raw}",
                        bits_tok.pos,
                    )
                turbo_bits = raw
            if self._peek().kind == TokenKind.ALWAYS:
                self._advance()
                self._expect(TokenKind.RAM)
                always_ram = True
            return QuantizationConfig(
                type=QuantizationType.TURBO,
                turbo_bits=turbo_bits,
                always_ram=always_ram,
            )

        raise QQLSyntaxError(
            f"Expected SCALAR, BINARY, PRODUCT, or TURBO after QUANTIZE, got '{tok.value}'",
            tok.pos,
        )

    def _parse_drop(self) -> DropCollectionStmt:
        self._expect(TokenKind.DROP)
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        return DropCollectionStmt(collection=collection)

    def _parse_show(self) -> ShowCollectionsStmt | ShowCollectionStmt:
        self._expect(TokenKind.SHOW)
        if self._peek().kind == TokenKind.COLLECTIONS:
            self._advance()
            return ShowCollectionsStmt()
        self._expect(TokenKind.COLLECTION)
        collection = self._parse_identifier()
        return ShowCollectionStmt(collection=collection)

    def _parse_scroll(self) -> ScrollStmt:
        self._expect(TokenKind.SCROLL)
        self._expect(TokenKind.FROM)
        collection = self._parse_identifier()

        query_filter: FilterExpr | None = None
        after: str | int | None = None

        if self._peek().kind == TokenKind.WHERE:
            self._advance()
            query_filter = self._parse_filter_expr()

        if self._peek().kind == TokenKind.AFTER:
            self._advance()
            after = self._parse_point_id_value("SCROLL AFTER")

        self._expect(TokenKind.LIMIT)
        limit = int(self._expect(TokenKind.INTEGER).value)

        return ScrollStmt(
            collection=collection,
            limit=limit,
            query_filter=query_filter,
            after=after,
        )

    def _parse_select(self) -> SelectStmt:
        self._expect(TokenKind.SELECT)
        self._expect(TokenKind.STAR)
        self._expect(TokenKind.FROM)
        collection = self._parse_identifier()
        self._expect(TokenKind.WHERE)
        self._expect(TokenKind.ID)
        self._expect(TokenKind.EQUALS)
        point_id = self._parse_point_id_value("SELECT")
        return SelectStmt(collection=collection, point_id=point_id)

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
        fusion: str | None = None
        sparse_only: bool = False
        sparse_model: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()  # consume USING
            if self._peek().kind == TokenKind.HYBRID:
                self._advance()  # consume HYBRID
                hybrid = True
                # Optional FUSION / DENSE MODEL / SPARSE MODEL sub-clauses, any order.
                while self._peek().kind in (TokenKind.FUSION, TokenKind.DENSE, TokenKind.SPARSE):
                    sub = self._advance()
                    if sub.kind == TokenKind.FUSION:
                        value_tok = self._expect(TokenKind.STRING)
                        fusion = value_tok.value.lower()
                        if fusion not in _HYBRID_FUSION_VALUES:
                            raise QQLSyntaxError(
                                f"Unsupported hybrid fusion '{value_tok.value}'; expected 'rrf' or 'dbsf'",
                                value_tok.pos,
                            )
                        continue
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
                    indexed_only=with_clause.indexed_only,
                    quantization=with_clause.quantization,
                    mmr_diversity=with_clause.mmr_diversity,
                    mmr_candidates=with_clause.mmr_candidates,
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
                    indexed_only=parsed_with.indexed_only or with_clause.indexed_only,
                    quantization=parsed_with.quantization or with_clause.quantization,
                    mmr_diversity=(
                        parsed_with.mmr_diversity
                        if parsed_with.mmr_diversity is not None
                        else with_clause.mmr_diversity
                    ),
                    mmr_candidates=parsed_with.mmr_candidates or with_clause.mmr_candidates,
                )
        group_by: str | None = None
        group_size: int = 3
        if self._peek().kind == TokenKind.GROUP:
            self._advance()  # consume GROUP
            self._expect(TokenKind.BY)
            group_by = self._parse_field_path()
            if rerank:
                raise QQLSyntaxError(
                    "GROUP BY and RERANK cannot be combined in the same SEARCH statement",
                    self._peek().pos,
                )
            if self._peek().kind == TokenKind.GROUP_SIZE:
                self._advance()  # consume GROUP_SIZE
                gs_tok = self._peek()
                group_size = int(self._expect(TokenKind.INTEGER).value)
                if group_size <= 0:
                    raise QQLSyntaxError(
                        f"GROUP_SIZE must be a positive integer, got {group_size}",
                        gs_tok.pos,
                    )
        return SearchStmt(
            collection=collection,
            query_text=query_text,
            limit=limit,
            model=model,
            hybrid=hybrid,
            fusion=fusion,
            sparse_only=sparse_only,
            sparse_model=sparse_model,
            query_filter=query_filter,
            rerank=rerank,
            rerank_model=rerank_model,
            with_clause=with_clause,
            group_by=group_by,
            group_size=group_size,
        )

    def _parse_recommend(self) -> RecommendStmt:
        self._expect(TokenKind.RECOMMEND)
        self._expect(TokenKind.FROM)
        collection = self._parse_identifier()
        self._expect(TokenKind.POSITIVE)
        self._expect(TokenKind.IDS)
        positive_ids = self._parse_point_id_list()

        negative_ids: tuple[str | int, ...] = ()
        if self._peek().kind == TokenKind.NEGATIVE:
            self._advance()
            self._expect(TokenKind.IDS)
            negative_ids = self._parse_point_id_list()

        strategy: str | None = None
        if self._peek().kind == TokenKind.STRATEGY:
            self._advance()
            strategy = self._expect(TokenKind.STRING).value

        lookup_from: tuple[str, str | None] | None = None
        if self._peek().kind == TokenKind.LOOKUP:
            self._advance()
            self._expect(TokenKind.FROM)
            lookup_collection = self._parse_identifier()
            lookup_vector: str | None = None
            if self._peek().kind == TokenKind.VECTOR:
                self._advance()
                lookup_vector = self._expect(TokenKind.STRING).value
            lookup_from = (lookup_collection, lookup_vector)

        using: str | None = None
        if self._peek().kind == TokenKind.USING:
            self._advance()
            using = self._expect(TokenKind.STRING).value

        self._expect(TokenKind.LIMIT)
        limit = int(self._expect(TokenKind.INTEGER).value)

        offset: int = 0
        if self._peek().kind == TokenKind.OFFSET:
            self._advance()
            offset = int(self._expect(TokenKind.INTEGER).value)

        score_threshold: float | None = None
        if self._peek().kind == TokenKind.SCORE:
            self._advance()
            self._expect(TokenKind.THRESHOLD)
            score_threshold = float(self._expect(TokenKind.FLOAT).value)

        query_filter: FilterExpr | None = None
        if self._peek().kind == TokenKind.WHERE:
            self._advance()
            query_filter = self._parse_filter_expr()

        with_clause: SearchWith | None = None
        if self._peek().kind == TokenKind.WITH:
            self._advance()
            with_clause = self._parse_with_clause()

        return RecommendStmt(
            collection=collection,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            limit=limit,
            strategy=strategy,
            query_filter=query_filter,
            offset=offset,
            score_threshold=score_threshold,
            with_clause=with_clause,
            lookup_from=lookup_from,
            using=using,
        )

    def _parse_delete(self) -> DeleteStmt:
        self._expect(TokenKind.DELETE)
        self._expect(TokenKind.FROM)
        collection = self._parse_identifier()
        self._expect(TokenKind.WHERE)
        if self._peek().kind == TokenKind.ID:
            self._advance()
            self._expect(TokenKind.EQUALS)
            point_id = self._parse_point_id_value("DELETE")
            return DeleteStmt(collection=collection, point_id=point_id)

        query_filter = self._parse_filter_expr()
        return DeleteStmt(collection=collection, query_filter=query_filter)

    def _parse_update(self) -> UpdateVectorStmt | UpdatePayloadStmt:
        """
        UPDATE <collection> SET VECTOR WHERE id = <id> [<vector>]
        UPDATE <collection> SET PAYLOAD WHERE id = <id> {<payload>}
        UPDATE <collection> SET PAYLOAD WHERE <filter> {<payload>}
        """
        self._expect(TokenKind.UPDATE)
        collection = self._parse_identifier()
        self._expect(TokenKind.SET)

        if self._peek().kind == TokenKind.VECTOR:
            self._advance()  # consume VECTOR
            self._expect(TokenKind.WHERE)
            self._expect(TokenKind.ID)
            self._expect(TokenKind.EQUALS)
            point_id = self._parse_point_id_value("UPDATE SET VECTOR")
            vector_val = self._parse_value()  # parses [...] list
            if not isinstance(vector_val, list):
                raise QQLSyntaxError(
                    "Expected a vector list [...] after point ID in UPDATE SET VECTOR",
                    self._peek().pos,
                )
            try:
                for v in vector_val:
                    if isinstance(v, bool):
                        raise QQLSyntaxError(
                            "Vector elements must be numeric floats; "
                            "boolean values are not allowed",
                            self._peek().pos,
                        )
                coerced = tuple(float(v) for v in vector_val)
            except (ValueError, TypeError) as exc:
                raise QQLSyntaxError(
                    f"Vector elements must be numeric; got invalid value: {exc}",
                    self._peek().pos,
                ) from exc
            return UpdateVectorStmt(
                collection=collection,
                point_id=point_id,
                vector=coerced,
            )

        if self._peek().kind == TokenKind.PAYLOAD:
            self._advance()  # consume PAYLOAD
            self._expect(TokenKind.WHERE)
            if self._peek().kind == TokenKind.ID:
                self._advance()  # consume ID
                self._expect(TokenKind.EQUALS)
                point_id = self._parse_point_id_value("UPDATE SET PAYLOAD")
                payload = self._parse_dict()
                return UpdatePayloadStmt(
                    collection=collection, point_id=point_id, payload=payload
                )
            query_filter = self._parse_filter_expr()
            payload = self._parse_dict()
            return UpdatePayloadStmt(
                collection=collection, query_filter=query_filter, payload=payload
            )

        tok = self._peek()
        raise QQLSyntaxError(
            f"Expected VECTOR or PAYLOAD after SET, got '{tok.value}'", tok.pos
        )

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
        if tok.kind == TokenKind.IDENTIFIER:
            self._advance()
            return tok.value
        # Allow bare keywords to serve as field names (e.g. score, limit),
        # but not filter operator keywords or literal tokens.
        if tok.kind not in {
            TokenKind.AND, TokenKind.OR, TokenKind.NOT,
            TokenKind.IN, TokenKind.BETWEEN, TokenKind.IS,
            TokenKind.NULL, TokenKind.EMPTY, TokenKind.MATCH,
            TokenKind.ANY, TokenKind.PHRASE,
            TokenKind.STRING, TokenKind.INTEGER, TokenKind.FLOAT,
            TokenKind.LPAREN, TokenKind.RPAREN,
            TokenKind.LBRACE, TokenKind.RBRACE,
            TokenKind.LBRACKET, TokenKind.RBRACKET,
            TokenKind.COMMA, TokenKind.COLON, TokenKind.EQUALS,
            TokenKind.NOT_EQUALS, TokenKind.GT, TokenKind.GTE,
            TokenKind.LT, TokenKind.LTE, TokenKind.EOF,
        }:
            self._advance()
            return tok.value
        raise QQLSyntaxError(
            f"Expected a field name, got '{tok.value}'", tok.pos
        )

    def _parse_literal(self) -> str | int | float | bool:
        """STRING | INTEGER | FLOAT | boolean"""
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
        if tok.kind == TokenKind.IDENTIFIER:
            upper = tok.value.upper()
            if upper == "TRUE":
                self._advance()
                return True
            if upper == "FALSE":
                self._advance()
                return False
        raise QQLSyntaxError(
            f"Expected a literal value (string, integer, float, or boolean), got '{tok.value}'",
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

    def _parse_literal_list(self) -> list[str | int | float | bool]:
        """'(' literal { ',' literal } [','] ')'  — used by IN / NOT IN."""
        self._expect(TokenKind.LPAREN)
        items: list[str | int | float | bool] = []
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

    def _parse_point_id_list(self) -> tuple[str | int, ...]:
        self._expect(TokenKind.LPAREN)
        items: list[str | int] = []
        if self._peek().kind == TokenKind.RPAREN:
            raise QQLSyntaxError("Expected at least one point id", self._peek().pos)
        while True:
            tok = self._peek()
            if tok.kind == TokenKind.STRING:
                self._advance()
                items.append(tok.value)
            elif tok.kind == TokenKind.INTEGER:
                self._advance()
                items.append(int(tok.value))
            else:
                raise QQLSyntaxError(
                    f"Expected string or integer point id, got '{tok.value}'",
                    tok.pos,
                )
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                if self._peek().kind == TokenKind.RPAREN:
                    break
            else:
                break
        self._expect(TokenKind.RPAREN)
        return tuple(items)

    def _parse_point_id_value(self, statement: str) -> str | int:
        tok = self._peek()
        if tok.kind == TokenKind.STRING:
            self._advance()
            return tok.value
        if tok.kind == TokenKind.INTEGER:
            self._advance()
            return int(tok.value)
        raise QQLSyntaxError(
            f"{statement} requires a string or integer point id, got '{tok.value}'",
            tok.pos,
        )

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

    # ── WITH clause: { hnsw_ef: N, exact: true, acorn: true, ... } ──

    def _parse_with_clause(self) -> SearchWith:
        self._expect(TokenKind.LBRACE)
        hnsw_ef: int | None = None
        exact: bool = False
        acorn: bool = False
        indexed_only: bool = False
        quantization: QuantizationSearchWith | None = None
        mmr_diversity: float | None = None
        mmr_candidates: int | None = None
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
            elif key == "indexed_only":
                indexed_only = self._parse_bool()
            elif key == "quantization":
                quantization = self._parse_quantization_search_with()
            elif key == "mmr_diversity":
                mmr_diversity = float(self._parse_number())
                if not 0.0 <= mmr_diversity <= 1.0:
                    raise QQLSyntaxError(
                        f"mmr_diversity must be between 0 and 1, got {mmr_diversity}",
                        key_tok.pos,
                    )
            elif key == "mmr_candidates":
                mmr_candidates = int(self._expect(TokenKind.INTEGER).value)
                if mmr_candidates <= 0:
                    raise QQLSyntaxError(
                        f"mmr_candidates must be a positive integer, got {mmr_candidates}",
                        key_tok.pos,
                    )
            else:
                raise QQLSyntaxError(
                    "Unknown WITH parameter "
                    f"'{key}'. Expected: hnsw_ef, exact, acorn, indexed_only, quantization, mmr_diversity, mmr_candidates",
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
            indexed_only=indexed_only,
            quantization=quantization,
            mmr_diversity=mmr_diversity,
            mmr_candidates=mmr_candidates,
        )

    def _parse_quantization_search_with(self) -> QuantizationSearchWith:
        self._expect(TokenKind.LBRACE)
        ignore: bool | None = None
        rescore: bool | None = None
        oversampling: float | None = None

        while self._peek().kind != TokenKind.RBRACE:
            key_tok = self._expect(TokenKind.IDENTIFIER)
            key = key_tok.value.lower()
            self._expect(TokenKind.COLON)
            if key == "ignore":
                ignore = self._parse_bool()
            elif key == "rescore":
                rescore = self._parse_bool()
            elif key == "oversampling":
                oversampling = float(self._parse_number())
            else:
                raise QQLSyntaxError(
                    "Unknown quantization parameter "
                    f"'{key}'. Expected: ignore, rescore, oversampling",
                    key_tok.pos,
                )
            if self._peek().kind == TokenKind.COMMA:
                self._advance()
                if self._peek().kind == TokenKind.RBRACE:
                    break
            else:
                break

        self._expect(TokenKind.RBRACE)
        return QuantizationSearchWith(
            ignore=ignore,
            rescore=rescore,
            oversampling=oversampling,
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
