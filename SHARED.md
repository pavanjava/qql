# Cross-Repository Documentation Notes

QQL has a Python implementation (`qql-cli`) and a companion Go implementation (`qql-go`). Some documentation is useful to keep conceptually aligned across both projects, but each repository remains responsible for its own implementation details.

This file is a coordination guide, not a hard synchronization system.

## Shared in Spirit

These documents should use the same terminology and avoid contradicting each other across implementations:

| File | Purpose |
|---|---|
| `ROADMAP.md` | Project direction and priority areas |
| `.github/RFC_TEMPLATE.md` | Template for syntax and behavior proposals |
| `.github/ISSUE_LABELS.md` | Suggested issue label taxonomy |
| `docs/CONTRIBUTING.md` | Contributor workflow |
| `docs/SYNTAX_GUIDELINES.md` | How to add or change QQL syntax |
| `docs/COMPATIBILITY.md` | Feature matrix across Qdrant, Python, and Go |
| `docs/RFCS/README.md` | RFC process overview |

## Repository-Specific

These files should normally stay different:

| Python `qql` | Go `qql-go` | Why |
|---|---|---|
| `README.md` | `README.md` | Different install, command, and release details |
| `pyproject.toml` | `go.mod` | Different package managers |
| `src/qql/` | Go source tree | Different implementations |
| `tests/` | Go tests | Different test frameworks |
| release notes | release notes | Different version history |

## Update Guidance

When a change affects the QQL language rather than one implementation:

1. Update the local documentation.
2. Note whether the behavior is Python-only, Go-only, or shared.
3. If the companion implementation is affected, open or link a tracking issue there.
4. Avoid blocking one implementation's documentation on the other unless the feature requires true lockstep behavior.

## Long-Term Options

If cross-repo drift becomes painful, consider one of these later:

- a small `qql-spec` repository for syntax and compatibility docs
- a CI check that compares selected docs between repos
- release notes that explicitly call out Python and Go parity gaps

For now, keep the process lightweight and accurate.
