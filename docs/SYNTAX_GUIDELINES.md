# Syntax Guidelines

Use this guide when adding or changing QQL syntax.

The important rule: syntax changes must update the whole path from lexer to docs. A statement is not complete just because the parser accepts it.

## Before Coding

Write down:

- the exact syntax
- the Qdrant API it maps to
- whether it needs embeddings
- whether it changes JSON output
- whether Go should eventually match
- known limitations

If the change adds a statement, changes existing syntax, or changes JSON output, write an RFC first.

## Implementation Checklist

For Python `qql-cli`, most syntax changes touch:

| Stage | Typical file |
|---|---|
| Lexer | `src/qql/lexer.py` |
| AST | `src/qql/ast_nodes.py` |
| Parser | `src/qql/parser.py` |
| Executor | `src/qql/executor.py` |
| Tests | `tests/test_lexer.py`, `tests/test_parser.py`, `tests/test_executor.py` |
| Docs | `docs/syntax/README.md`, `docs/COMPATIBILITY.md`, maybe `README.md` |

For Go parity, use the equivalent lexer/parser/AST/executor/test locations in `qql-go`. A Python PR can include a Go parity note without implementing the Go side.

## Example: Adding `COUNT`

Proposed syntax:

```sql
COUNT <collection>
COUNT <collection> WHERE <filter>
```

Qdrant mapping:

| QQL | Qdrant |
|---|---|
| `COUNT <collection>` | `client.count(collection_name=...)` |
| `WHERE <filter>` | `count_filter` |

Expected output:

```text
42 point(s) in 'articles'
```

Expected JSON shape:

```json
{
  "success": true,
  "message": "42 point(s) in 'articles'",
  "data": {
    "count": 42
  }
}
```

## Parser Rules

Keep grammar small and predictable:

- prefer one obvious clause order
- avoid aliases unless there is a compatibility reason
- reuse existing filter parsing where possible
- reject unsupported syntax with clear errors
- do not silently ignore extra tokens

For `COUNT`, a simple grammar is enough:

```text
count_stmt := "COUNT" identifier [where_clause]
```

## Executor Rules

Executor code should:

- check collection existence when needed
- convert QQL filters through the existing filter builder
- call the closest Qdrant API directly
- return a stable `ExecutionResult`
- wrap Qdrant errors with a QQL-specific message

Do not add planner or optimizer layers unless the feature genuinely needs them.

## Tests

At minimum, add tests for:

- lexer recognizes new keywords
- parser builds the right AST
- invalid syntax fails clearly
- executor calls the expected Qdrant client method
- JSON/human output stays stable if the CLI formats the result

If the feature maps to Qdrant behavior that is hard to mock, add a small integration test separately and document the setup.

## Documentation

Update:

- `docs/COMPATIBILITY.md`
- `docs/syntax/README.md`
- root `README.md` only when the feature is stable and important for most users

Dedicated syntax pages are welcome, but only link to them after they exist.

Each syntax page should include:

- syntax
- 2-3 examples
- output shape
- limitations
- version/support notes

## Common Pitfalls

- Adding parser support without executor support.
- Updating Python docs while forgetting Go parity notes.
- Documenting planned syntax as implemented.
- Returning a JSON shape that differs from similar statements.
- Adding broad syntax when a smaller first version would solve the use case.
