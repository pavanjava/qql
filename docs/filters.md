# WHERE Clause Filters

The `WHERE` clause lets you filter on any payload field using SQL-style predicates. All standard comparison, range, membership, null-check, and full-text operators are supported.

`WHERE` works on `SEARCH`, `RECOMMEND`, and `DELETE` statements.

---

## Equality and inequality

```sql
-- Exact match
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE category = 'paper'

-- Boolean match
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE active = true

-- Not equal
SEARCH articles SIMILAR TO 'ml' LIMIT 10 WHERE status != 'draft'
```

---

## Range comparisons

```sql
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE score > 0.8
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE year < 2024
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE score >= 0.75
SEARCH articles SIMILAR TO 'ai' LIMIT 5 WHERE year <= 2023
```

---

## BETWEEN … AND

```sql
-- Inclusive range (equivalent to year >= 2018 AND year <= 2023)
SEARCH articles SIMILAR TO 'history of ai' LIMIT 10 WHERE year BETWEEN 2018 AND 2023
```

---

## IN and NOT IN

```sql
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE status IN ('published', 'reviewed')
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE status NOT IN ('deleted', 'archived')
SEARCH articles SIMILAR TO 'retrieval' LIMIT 10 WHERE active IN (true, false)
```

---

## IS NULL and IS NOT NULL

```sql
SEARCH articles SIMILAR TO 'peer review' LIMIT 5 WHERE reviewer IS NULL
SEARCH articles SIMILAR TO 'peer review' LIMIT 5 WHERE reviewer IS NOT NULL
```

---

## IS EMPTY and IS NOT EMPTY

```sql
SEARCH articles SIMILAR TO 'untagged' LIMIT 5 WHERE tags IS EMPTY
SEARCH articles SIMILAR TO 'categorized' LIMIT 5 WHERE tags IS NOT EMPTY
```

---

## Full-text MATCH

```sql
-- All terms must appear in the field (requires a Qdrant full-text index)
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH 'vector database'

-- Any term can match
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH ANY 'embedding retrieval'

-- Exact phrase must appear
SEARCH articles SIMILAR TO 'search' LIMIT 10 WHERE title MATCH PHRASE 'semantic search'
```

> To use `MATCH` operators efficiently, create a full-text index first:
> ```sql
> CREATE INDEX ON COLLECTION articles FOR title TYPE text
> ```

---

## AND, OR, NOT — logical operators

Operator precedence: `NOT` (highest) > `AND` > `OR` (lowest). Use parentheses to override.

```sql
-- AND: both conditions must be true
SEARCH articles SIMILAR TO 'nlp' LIMIT 10 WHERE category = 'paper' AND year >= 2020

-- OR: either condition can be true
SEARCH articles SIMILAR TO 'llm' LIMIT 10 WHERE source = 'arxiv' OR source = 'pubmed'

-- NOT: negate a condition
SEARCH articles SIMILAR TO 'benchmark' LIMIT 10 WHERE NOT status = 'draft'

-- Parentheses to group OR inside AND
SEARCH articles SIMILAR TO 'conference paper' LIMIT 10
  WHERE (source = 'arxiv' OR source = 'ieee') AND year >= 2022

-- NOT on a parenthesized group
SEARCH articles SIMILAR TO 'x' LIMIT 5 WHERE NOT (status = 'draft' OR status = 'deleted')
```

---

## Dot-notation for nested fields

```sql
SEARCH articles SIMILAR TO 'wikipedia' LIMIT 5 WHERE meta.source = 'web'
SEARCH cities SIMILAR TO 'large city' LIMIT 5 WHERE country.cities[].population > 1000000
```

---

## WHERE also works in hybrid mode

```sql
SEARCH articles SIMILAR TO 'deep learning' LIMIT 10
  USING HYBRID WHERE year BETWEEN 2020 AND 2024 AND status = 'published'
```

---

## WHERE in DELETE

```sql
-- Delete by filter
DELETE FROM articles WHERE category = 'archived'

-- Delete with compound filter
DELETE FROM articles WHERE year < 2020 AND status = 'draft'
```

---

## Full filter reference

| WHERE syntax | Description |
|---|---|
| `field = 'x'` | Exact match |
| `field != 'x'` | Not equal |
| `field > n` | Greater than |
| `field >= n` | Greater than or equal |
| `field < n` | Less than |
| `field <= n` | Less than or equal |
| `field BETWEEN a AND b` | Inclusive range |
| `field IN ('a', 'b')` | Value in list |
| `field NOT IN ('a', 'b')` | Value not in list |
| `field IS NULL` | Field absent or null |
| `field IS NOT NULL` | Field present and non-null |
| `field IS EMPTY` | Field is an empty list |
| `field IS NOT EMPTY` | Field is a non-empty list |
| `field MATCH 'text'` | All terms present (full-text) |
| `field MATCH ANY 'text'` | Any term present (full-text) |
| `field MATCH PHRASE 'text'` | Exact phrase present (full-text) |
| `A AND B` | Both conditions must hold |
| `A OR B` | Either condition must hold |
| `NOT A` | Condition must not hold |
| `(A OR B) AND C` | Parentheses for grouping |
| `meta.source = 'x'` | Dot-notation nested field |
