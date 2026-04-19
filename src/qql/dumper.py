"""QQL collection dumper — exports a Qdrant collection to a .qql script file.

The generated file contains:
  1. A header comment with metadata
  2. CREATE COLLECTION <name> [HYBRID]
  3. One INSERT BULK statement per batch of _DUMP_BATCH_SIZE points
  4. A footer comment with totals

The file is valid QQL and can be re-executed with ``qql execute <file>``.
Points that lack a ``'text'`` payload field are skipped (with a warning
comment written into the file).
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rich.console import Console

_DUMP_BATCH_SIZE = 50


# ── Value serializer ──────────────────────────────────────────────────────────


def _serialize_value(v: Any) -> str:
    """Recursively convert a Python payload value to valid QQL syntax."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(v, list):
        items = ", ".join(_serialize_value(i) for i in v)
        return f"[{items}]"
    if isinstance(v, dict):
        return _serialize_dict(v, indent=4)
    # Fallback: stringify
    return f"'{v}'"


def _serialize_dict(d: dict[str, Any], indent: int = 4) -> str:
    """Serialize a dict to a multi-line QQL ``{...}`` block."""
    pad = " " * indent
    lines = ["{"]
    items = list(d.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"{pad}'{key}': {_serialize_value(value)}{comma}")
    lines.append("}")
    return "\n".join(lines)


# ── Collection type detection ─────────────────────────────────────────────────


def _is_hybrid(collection: str, client: QdrantClient) -> bool:
    """Return True if the collection uses named vectors (dense + sparse)."""
    info = client.get_collection(collection)
    vectors = info.config.params.vectors  # type: ignore[union-attr]
    return isinstance(vectors, dict)


# ── Main entry point ──────────────────────────────────────────────────────────


def dump_collection(
    collection: str,
    output_path: str,
    client: QdrantClient,
    console: Console,
    err_console: Console,
) -> tuple[int, int]:
    """Export every point in *collection* to a .qql script at *output_path*.

    Returns ``(points_written, points_skipped)`` counts.
    Points without a ``'text'`` key are skipped and counted in *points_skipped*.
    """
    if not client.collection_exists(collection):
        err_console.print(
            f"[bold red]Error:[/bold red] Collection '{collection}' does not exist."
        )
        return 0, 0

    hybrid = _is_hybrid(collection, client)
    col_type = "hybrid (dense + sparse)" if hybrid else "dense"
    using_clause = " USING HYBRID" if hybrid else ""

    # ── First pass: count total points for the header ─────────────────────
    count_info = client.count(collection_name=collection, exact=True)
    total_points = count_info.count
    total_batches = max(1, math.ceil(total_points / _DUMP_BATCH_SIZE))

    console.print(
        f"  Collection type : [cyan]{col_type}[/cyan]\n"
        f"  Points          : [cyan]{total_points}[/cyan]\n"
        f"  Batches         : [cyan]{total_batches}[/cyan] "
        f"([dim]{_DUMP_BATCH_SIZE} points/batch[/dim])\n"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    batch_num = 0

    with out.open("w", encoding="utf-8") as f:
        # ── Header comment ────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"-- ============================================================\n"
            f"-- QQL Dump — collection: {collection}\n"
            f"-- Generated : {ts}\n"
            f"-- Points    : {total_points}\n"
            f"-- Type      : {col_type}\n"
            f"-- Note      : Re-importing re-embeds all text using the\n"
            f"--             configured model (see: qql connect).\n"
            f"-- ============================================================\n"
            f"\n"
        )

        # ── CREATE statement ──────────────────────────────────────────────
        hybrid_suffix = " HYBRID" if hybrid else ""
        f.write(f"CREATE COLLECTION {collection}{hybrid_suffix}\n\n")

        # ── Paginate and write INSERT BULK batches ────────────────────────
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=_DUMP_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            batch_num += 1
            batch_start = (batch_num - 1) * _DUMP_BATCH_SIZE + 1
            batch_end = batch_start + len(records) - 1

            # Filter points that have a 'text' field
            valid = []
            for rec in records:
                payload = rec.payload or {}
                if "text" not in payload:
                    skipped += 1
                    continue
                valid.append(payload)

            if valid:
                f.write(
                    f"-- Batch {batch_num} / {total_batches}"
                    f"  (records {batch_start}–{batch_end})\n"
                )
                f.write(
                    f"INSERT BULK INTO COLLECTION {collection} VALUES [\n"
                )
                for i, payload in enumerate(valid):
                    dict_str = _serialize_dict(payload, indent=4)
                    # Indent the entire dict block by 2 spaces
                    indented = "\n".join(
                        "  " + line for line in dict_str.splitlines()
                    )
                    comma = "," if i < len(valid) - 1 else ""
                    f.write(f"{indented}{comma}\n")
                    written += 1
                f.write(f"]{using_clause}\n\n")
            else:
                # All records in this batch were skipped
                f.write(
                    f"-- Batch {batch_num} / {total_batches}"
                    f"  (records {batch_start}–{batch_end})"
                    f" — all skipped (no 'text' field)\n\n"
                )

            console.print(
                f"  [dim][[{batch_num}/{total_batches}]][/dim] "
                f"wrote {len(valid)} point(s)"
                + (f", skipped {len(records) - len(valid)}" if len(records) != len(valid) else "")
            )

            if next_offset is None:
                break
            offset = next_offset

        # ── Footer comment ────────────────────────────────────────────────
        f.write(
            f"-- ============================================================\n"
            f"-- End of dump\n"
            f"-- Written : {written}\n"
            f"-- Skipped : {skipped}  (no 'text' field)\n"
            f"-- ============================================================\n"
        )

    return written, skipped
