# Contributing to QQL

Thanks for helping improve QQL. This project is small enough that contributions should stay focused: fix one behavior, add one feature, or improve one document at a time.

## Project Scope

This repository contains the Python implementation, `qql-cli`.

There is also a companion Go implementation, `qql-go`. Parity is useful and should be tracked, but a Python contribution does not need to implement the Go side in the same PR unless the maintainers explicitly ask for it.

## Getting Started

1. Read the root `README.md` for current user-facing behavior.
2. Check `ROADMAP.md` for likely priorities.
3. For syntax or behavior changes, check whether an issue or RFC already exists.
4. Keep the first PR small.

## Development Setup

```bash
git clone https://github.com/pavanjava/qql.git
cd qql
uv sync
uv run pytest tests/ -v
```

If you do not use `uv`, install the package in editable mode with your preferred Python workflow and run `pytest`.

## Reporting Bugs

Before opening a bug report:

1. Check existing issues.
2. Test against the latest available version.
3. Reduce the problem to the smallest `.qql` statement or script you can.

Include:

- QQL version
- Python version
- Qdrant version and deployment type
- the exact QQL command or script
- expected output
- actual output or error
- whether the issue appears in human output, JSON output, or both

## Requesting Features

Feature requests usually fall into two groups:

| Type | Meaning |
|---|---|
| Qdrant parity | Qdrant supports the operation but QQL does not expose it yet |
| QQL enhancement | A QQL-specific convenience or workflow |

Open a normal issue for small Qdrant parity gaps. Use an RFC for syntax changes, CLI surface changes, JSON contract changes, or anything likely to affect both Python and Go.

## Pull Requests

Use this checklist:

- keep the change scoped
- add or update tests for code changes
- update docs when syntax or behavior changes
- avoid unrelated formatting/refactors
- explain how you verified the change

Suggested PR body:

```markdown
## Summary
One sentence describing the change.

## Changes
- What changed
- What was intentionally left out

## Testing
- Command(s) run

## Notes
Any compatibility, migration, or Go parity notes.
```

## When an RFC Is Required

Use an RFC for:

- a new statement type, such as `COUNT` or `UPDATE`
- changes to existing syntax
- new CLI commands or flags
- changes to JSON output shape
- breaking changes
- new inference or embedding modes

An RFC is not required for:

- bug fixes
- documentation-only changes
- tests
- internal refactors with no user-visible behavior change
- examples or tutorials

## Coding Standards

### Python

- Follow the existing style.
- Keep parser, AST, executor, and tests in sync.
- Prefer clear error messages over generic exceptions.
- Keep AST dataclasses immutable unless there is a strong reason not to.
- Do not add abstractions for one-off code.

### Go Parity Notes

When a Python change affects shared QQL syntax, add a short note in the PR about Go impact:

- no Go impact
- Go should eventually match
- Go behavior is intentionally different
- Go status unknown

This is enough for the Python PR unless maintainers request a coordinated change.

## Parser / Lexer Changes

For new syntax, update the full pipeline:

1. Lexer token or keyword
2. AST node
3. Parser rule
4. Executor behavior
5. Unit tests
6. Documentation
7. Compatibility matrix

See `docs/SYNTAX_GUIDELINES.md` for a longer walkthrough.

## Testing

Run:

```bash
uv run pytest tests/ -v
```

Integration tests, if added, should clearly document their Qdrant setup requirements.

## Documentation

Keep docs accurate before making them broad.

| File | Purpose |
|---|---|
| `README.md` | Main user entry point |
| `ROADMAP.md` | Priorities and planned direction |
| `docs/syntax/` | Syntax reference |
| `docs/COMPATIBILITY.md` | Feature support matrix |
| `docs/RFCS/` | Design proposals |
| `docs/TUTORIALS/` | Runnable workflows |
| `docs/MIGRATING/` | Migration notes |

Do not link to missing pages as if they already exist. Mark planned docs as planned until they are written.

## Release Process

Maintainers handle versioning and releases. Contributors usually do not need to bump versions.

## Questions

Use GitHub Discussions for open-ended questions and issues for scoped bugs or feature requests.
