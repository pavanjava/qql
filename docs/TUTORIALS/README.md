# Tutorials

This directory is for runnable, end-to-end QQL workflows.

At the moment it is an index and template. Add tutorial files only after the commands have been tested against a real Qdrant instance.

## Good First Tutorials

| Tutorial | Goal | Status |
|---|---|---|
| `01-quick-start.md` | Create a collection, insert data, search, clean up | Planned |
| `02-filtered-search.md` | Use `WHERE` filters and payload indexes | Planned |
| `03-hybrid-search.md` | Compare dense, sparse, and hybrid search | Planned |
| `04-reranking.md` | Show when `RERANK` improves precision | Planned |
| `05-dump-restore.md` | Export and restore a small collection | Planned |

## Tutorial Rules

- Keep one tutorial focused on one workflow.
- Include setup and cleanup.
- Use small sample data.
- Show expected output when it helps.
- Avoid domain-heavy examples until the basics are covered.
- Do not depend on private services or credentials.

## Template

Use this structure for new tutorial files:

### Title

Short, task-oriented name.

> Time: 5-10 minutes
> Requires: Qdrant running locally

### Goal

What the user will accomplish.

### Setup

```bash
docker run -p 6333:6333 qdrant/qdrant
qql connect --url http://localhost:6333
```

### Steps

```sql
CREATE COLLECTION demo
INSERT INTO COLLECTION demo VALUES {'text': 'hello vector search'}
SEARCH demo SIMILAR TO 'hello' LIMIT 3
```

### Expected Result

Describe the important output, not every character.

### Cleanup

```sql
DROP COLLECTION demo
```

### Next

Link to related docs or tutorials.
