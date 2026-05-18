"""QQL collection dumper — exports a Qdrant collection to a .qql script file.

The generated file contains:
  1. A header comment with metadata
  2. CREATE COLLECTION <name> [HYBRID]
  3. One INSERT BULK statement per batch of *batch_size* points
     (default _DEFAULT_DUMP_BATCH_SIZE = 50, overridable via the CLI flag)
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

_DEFAULT_DUMP_BATCH_SIZE = 50


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


def _collection_info(collection: str, client: QdrantClient) -> Any:
    return client.get_collection(collection)


def _is_hybrid(collection: str, client: QdrantClient) -> bool:
    """Return True only when sparse vectors are configured."""
    info = _collection_info(collection, client)
    sparse_vectors = info.config.params.sparse_vectors
    return isinstance(sparse_vectors, dict) and bool(sparse_vectors)


def _quantization_clause(info: Any) -> str:
    quant = info.config.quantization_config
    if quant is None:
        return ""
    if hasattr(quant, "scalar"):
        clause = " QUANTIZE SCALAR"
        if quant.scalar.quantile is not None:
            clause += f" QUANTILE {quant.scalar.quantile}"
        if quant.scalar.always_ram:
            clause += " ALWAYS RAM"
        return clause
    if hasattr(quant, "binary"):
        clause = " QUANTIZE BINARY"
        if quant.binary.always_ram:
            clause += " ALWAYS RAM"
        return clause
    if hasattr(quant, "product"):
        clause = " QUANTIZE PRODUCT"
        if quant.product.always_ram:
            clause += " ALWAYS RAM"
        return clause
    if hasattr(quant, "turbo"):
        clause = " QUANTIZE TURBO"
        bits = quant.turbo.bits
        if bits is not None:
            bit_map = {
                "BITS4": "4",
                "BITS2": "2",
                "BITS1_5": "1.5",
                "BITS1": "1",
            }
            clause += f" BITS {bit_map.get(getattr(bits, 'name', ''), str(bits))}"
        if quant.turbo.always_ram:
            clause += " ALWAYS RAM"
        return clause
    return ""


def _config_clauses(info: Any) -> str:
    clauses: list[str] = []
    params = info.config.params
    vectors = params.vectors  # type: ignore[union-attr]
    dense_vectors = vectors.get("dense") if isinstance(vectors, dict) else vectors
    if dense_vectors is not None and getattr(dense_vectors, "on_disk", None) is not None:
        clauses.append(f"WITH VECTORS {{ on_disk: {'true' if dense_vectors.on_disk else 'false'} }}")

    hnsw = info.config.hnsw_config
    hnsw_items: list[str] = []
    for key in (
        "m",
        "ef_construct",
        "full_scan_threshold",
        "max_indexing_threads",
        "payload_m",
    ):
        value = getattr(hnsw, key, None)
        if value is not None:
            hnsw_items.append(f"{key}: {value}")
    for key in ("on_disk", "inline_storage"):
        value = getattr(hnsw, key, None)
        if value is not None:
            hnsw_items.append(f"{key}: {'true' if value else 'false'}")
    if hnsw_items:
        clauses.append(f"WITH HNSW {{ {', '.join(hnsw_items)} }}")

    optimizers = getattr(info.config, "optimizer_config", None) or getattr(info.config, "optimizers_config", None)
    optimizer_items: list[str] = []
    if optimizers is not None:
        for key in (
            "deleted_threshold",
            "vacuum_min_vector_number",
            "default_segment_number",
            "max_segment_size",
            "memmap_threshold",
            "indexing_threshold",
            "flush_interval_sec",
        ):
            value = getattr(optimizers, key, None)
            if value is not None:
                optimizer_items.append(f"{key}: {value}")
        max_opt_threads = getattr(optimizers, "max_optimization_threads", None)
        if max_opt_threads is not None:
            value = getattr(max_opt_threads, "value", max_opt_threads)
            optimizer_items.append(f"max_optimization_threads: {value}")
        prevent_unoptimized = getattr(optimizers, "prevent_unoptimized", None)
        if prevent_unoptimized is not None:
            optimizer_items.append(
                f"prevent_unoptimized: {'true' if prevent_unoptimized else 'false'}"
            )
    if optimizer_items:
        clauses.append(f"WITH OPTIMIZERS {{ {', '.join(optimizer_items)} }}")

    param_items: list[str] = []
    for key in ("replication_factor", "write_consistency_factor", "on_disk_payload"):
        value = getattr(params, key, None)
        if value is not None:
            if isinstance(value, bool):
                param_items.append(f"{key}: {'true' if value else 'false'}")
            else:
                param_items.append(f"{key}: {value}")
    if param_items:
        clauses.append(f"WITH PARAMS {{ {', '.join(param_items)} }}")
    return (" " + " ".join(clauses)) if clauses else ""


# ── Main entry point ──────────────────────────────────────────────────────────


def dump_collection(
    collection: str,
    output_path: str,
    client: QdrantClient,
    console: Console,
    err_console: Console,
    batch_size: int = _DEFAULT_DUMP_BATCH_SIZE,
) -> tuple[int, int]:
    """Export every point in *collection* to a .qql script at *output_path*.

    Returns ``(points_written, points_skipped)`` counts.
    Points without a ``'text'`` key are skipped and counted in *points_skipped*.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    if not client.collection_exists(collection):
        err_console.print(
            f"[bold red]Error:[/bold red] Collection '{collection}' does not exist."
        )
        return 0, 0

    info = _collection_info(collection, client)
    sparse_vectors = info.config.params.sparse_vectors
    hybrid = isinstance(sparse_vectors, dict) and bool(sparse_vectors)
    col_type = "hybrid (dense + sparse)" if hybrid else "dense"
    using_clause = " USING HYBRID" if hybrid else ""

    # ── First pass: count total points for the header ─────────────────────
    count_info = client.count(collection_name=collection, exact=True)
    total_points = count_info.count
    total_batches = max(1, math.ceil(total_points / batch_size))

    console.print(
        f"  Collection type : [cyan]{col_type}[/cyan]\n"
        f"  Points          : [cyan]{total_points}[/cyan]\n"
        f"  Batches         : [cyan]{total_batches}[/cyan] "
        f"([dim]{batch_size} points/batch[/dim])\n"
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
        config_suffix = _config_clauses(info)
        quantization_suffix = _quantization_clause(info)
        f.write(
            f"CREATE COLLECTION {collection}{hybrid_suffix}{config_suffix}{quantization_suffix}\n\n"
        )

        # ── Paginate and write INSERT BULK batches ────────────────────────
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            batch_num += 1
            batch_start = (batch_num - 1) * batch_size + 1
            batch_end = batch_start + len(records) - 1

            # Filter points that have a 'text' field
            valid = []
            for rec in records:
                payload = rec.payload or {}
                if "text" not in payload:
                    skipped += 1
                    continue
                dump_payload = dict(payload)
                dump_payload["id"] = rec.id
                valid.append(dump_payload)

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
