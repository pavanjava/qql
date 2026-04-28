# RFC: <Short Title>

- **Status:** Draft / Proposed / Accepted / Rejected / Implemented
- **Author:** @github-username
- **Target:** Python / Go / Both
- **Created:** YYYY-MM-DD

## Summary

Describe the proposed change in one paragraph.

## Motivation

What problem does this solve?

- Current pain:
- Who is affected:
- Why QQL should expose this:

## Proposed Syntax

```sql
-- Minimal example
NEW SYNTAX ...

-- Example with optional clauses
NEW SYNTAX ... WHERE ... WITH { ... }
```

## Qdrant Mapping

| QQL syntax | Qdrant API/model |
|---|---|
| `...` | `...` |

If Qdrant does not directly support the behavior, explain why QQL should still add it.

## Output

Human-readable output:

```text
...
```

JSON output:

```json
{
  "success": true,
  "message": "...",
  "data": {}
}
```

## Compatibility

- Does this break existing QQL scripts?
- Does this affect JSON output contracts?
- Should Python and Go match?
- Can one implementation ship first?

## Implementation Plan

1. Lexer/parser/AST changes
2. Executor changes
3. Tests
4. Documentation

## Alternatives

List simpler or competing designs and why they were not chosen.

## Open Questions

- Question 1
- Question 2

## References

- Qdrant docs:
- Related issues:
- Related PRs:
