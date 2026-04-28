# RFCs

RFCs are lightweight design notes for changes that affect QQL syntax, CLI behavior, JSON output, or cross-implementation compatibility.

Use an RFC to slow down decisions that would be hard to reverse. Do not use RFCs for routine bug fixes or documentation cleanup.

## When to Write an RFC

Write an RFC for:

- a new statement type, such as `GET`, `COUNT`, `SCROLL`, or `UPDATE`
- changes to existing statement syntax
- new CLI commands or flags
- JSON output contract changes
- breaking changes
- new embedding or inference modes

No RFC is needed for:

- bug fixes
- tests
- documentation updates
- examples and tutorials
- internal refactors with no user-visible behavior change

## Process

1. Copy `.github/RFC_TEMPLATE.md`.
2. Create `docs/RFCS/NNNN-short-title.md`.
3. Mark the status as `Draft`.
4. Open a PR when ready for discussion.
5. Update the RFC as decisions are made.
6. Link implementation PRs after merge.

## Statuses

| Status | Meaning |
|---|---|
| Draft | The author is still shaping the proposal |
| Proposed | Ready for maintainer/community review |
| Accepted | Approved for implementation |
| Rejected | Declined, with reason documented |
| Implemented | Merged into at least one implementation |

## RFC Index

No RFCs have been accepted yet.

When RFCs are added, keep the index grouped by status:

| RFC | Title | Status | Implementation |
|---|---|---|---|
| `0001-example.md` | Example title | Proposed | Not implemented |

## Guidance

Good RFCs are specific:

- show exact syntax
- map syntax to Qdrant APIs
- define human-readable and JSON output
- list known limitations
- state whether Python and Go should match
- explain alternatives considered

Keep one RFC focused on one behavior. Large bundles are harder to review and easier to stall.
