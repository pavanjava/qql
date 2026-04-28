# QQL Roadmap

> **Status:** Draft for maintainer and community discussion  
> **Scope:** Public direction for the Python `qql-cli` project, with companion notes for the Go implementation where parity matters  
> **Principle:** Keep QQL small, readable, and close to Qdrant's real API surface

QQL is a SQL-like language and CLI for common Qdrant workflows. The near-term goal is not to cover every Qdrant feature. The goal is to make everyday vector database work easier to read, script, test, and share.

## Current Position

The Python implementation already supports the core workflow:

| Area | Status |
|---|---|
| Collection create/drop/list | Supported |
| Payload indexes | Supported |
| Insert and bulk insert | Supported |
| Dense search | Supported |
| Hybrid dense+sparse search | Supported |
| Sparse-only search | Supported |
| WHERE filters | Supported |
| Recommend by example IDs | Supported |
| Query-time search params | Supported |
| Reranking | Supported in Python |
| Delete by ID or filter | Supported |
| Script execution and dump/restore | Supported |
| Programmatic Python API | Supported through `run_query()` |

The Go implementation is developed separately. It should aim for language and behavior parity where practical, but this Python repository should not block on Go work before improving its own CLI and documentation.

## Near-Term Priorities

These are the best candidates for small, useful contributions. Each one should have tests and documentation before being considered complete.

| Priority | Feature | Why it matters | Suggested syntax |
|---|---|---|---|
| P0 | Get point by ID | Basic inspection is currently missing | `GET FROM <collection> WHERE id = '<id>'` |
| P0 | Scroll points | Needed for real datasets and exports | `SCROLL <collection> LIMIT 100` |
| P0 | Search pagination | Needed for browsing result sets | `SEARCH ... LIMIT 10 OFFSET 20` |
| P1 | Count points | Useful for validation and scripts | `COUNT <collection> WHERE <filter>` |
| P1 | Describe collection | Improves introspection and debugging | `DESCRIBE COLLECTION <name>` |
| P1 | Update payload | Avoids full reinsert for metadata changes | `UPDATE <collection> SET {...} WHERE id = '<id>'` |
| P1 | Delete payload keys | Removes fields without deleting points | `DELETE PAYLOAD field FROM <collection> WHERE id = '<id>'` |

## Later Ideas

These are worth exploring, but they should not distract from the smaller parity gaps above.

| Area | Possible work |
|---|---|
| Retrieval quality | MMR, score boosting, named vector search, batch search |
| Collection configuration | Distance selection, HNSW config, quantization, on-disk payload |
| Developer experience | Connection profiles, clearer JSON contracts, better error messages |
| Ecosystem | Syntax highlighting, examples, tutorials, migration guides |
| Operations | Collection aliases, snapshots, backup/restore workflows |

## Contribution Process

Use an RFC when a change affects syntax, CLI behavior, or JSON output. Small documentation fixes, tests, and bug fixes do not need an RFC.

Good roadmap issues should include:

- the Qdrant API being exposed
- the proposed QQL syntax
- expected human-readable output
- expected JSON output
- Python tests required
- Go parity notes, if relevant

## Documentation Goals

The documentation should stay practical:

- README: quick start and common usage
- `docs/syntax/`: compact syntax reference
- `docs/COMPATIBILITY.md`: checked feature matrix
- `docs/CONTRIBUTING.md`: contributor workflow
- `docs/RFCS/`: proposed and accepted syntax decisions
- `docs/TUTORIALS/`: runnable examples as they are added
- `docs/MIGRATING/`: focused migration notes

## Success Criteria

QQL is moving in the right direction when:

- users can inspect, insert, search, recommend, update, count, and export without dropping to raw SDK calls for common cases
- syntax changes are discussed before implementation
- docs describe what is implemented today, not only what is planned
- Python and Go differences are visible and intentional
- contributors can find small, well-scoped issues

This roadmap is intentionally modest. It should be revised as maintainers and contributors agree on scope.
