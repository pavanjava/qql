from __future__ import annotations

import sys

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from .config import delete_config, load_config, save_config, QQLConfig
from .exceptions import QQLError
from .executor import Executor
from .lexer import Lexer
from .parser import Parser

console = Console()
err_console = Console(stderr=True)

HELP_TEXT = """
[bold cyan]QQL — Qdrant Query Language[/bold cyan]

Available statements:

  [yellow]INSERT INTO COLLECTION[/yellow] <name> [yellow]VALUES[/yellow] {[yellow]'text'[/yellow]: '...', ...}
      Insert a point. 'text' is required and auto-vectorized.
      Optional: include [yellow]'id'[/yellow] in VALUES as an integer or UUID
      Optional: [yellow]USING MODEL[/yellow] '<model>'
      Optional: [yellow]USING HYBRID[/yellow] [DENSE MODEL '<model>'] [SPARSE MODEL '<model>']

  [yellow]INSERT BULK INTO COLLECTION[/yellow] <name> [yellow]VALUES[/yellow] [{[yellow]'text'[/yellow]: '...', ...}, ...]
      Batch insert multiple points in a single call. Each dict must contain 'text'.
      Optional: each dict may include [yellow]'id'[/yellow] as an integer or UUID.
      Supports the same [yellow]USING[/yellow] clauses as INSERT.

  [yellow]CREATE COLLECTION[/yellow] <name> [[yellow]HYBRID[/yellow]]
      Create a new collection. Add HYBRID for dense+sparse BM25 vectors.
      Optional: [yellow]USING MODEL[/yellow] '<model>'
      Optional: [yellow]USING HYBRID[/yellow] [DENSE MODEL '<model>']
      Optional: [yellow]QUANTIZE SCALAR[/yellow] [QUANTILE <0.0–1.0>] [ALWAYS RAM]
      Optional: [yellow]QUANTIZE BINARY[/yellow] [ALWAYS RAM]
      Optional: [yellow]QUANTIZE PRODUCT[/yellow] [ALWAYS RAM]   (4× compression)
      QUANTIZE may be combined with any HYBRID or MODEL clause.

  [yellow]DROP COLLECTION[/yellow] <name>
      Delete a collection and all its points.

  [yellow]SHOW COLLECTIONS[/yellow]
      List all collections in the connected Qdrant instance.

  [yellow]SHOW COLLECTION[/yellow] <name>
      Show detailed diagnostics for a single collection: point count, vector
      config, distance metric, quantization, HNSW parameters, payload indexes,
      and sharding info.

  [yellow]SCROLL FROM[/yellow] <name> [yellow]LIMIT[/yellow] <n>
      Paginate points by ID order.
      Optional: [yellow]WHERE[/yellow] <filter>
      Optional: [yellow]AFTER[/yellow] '<id>'|<int>

  [yellow]SELECT * FROM[/yellow] <name> [yellow]WHERE id =[/yellow] '<id>'|<int>
      Retrieve a single point by its ID and return its payload.

  [yellow]SEARCH[/yellow] <name> [yellow]SIMILAR TO[/yellow] '<text>' [yellow]LIMIT[/yellow] <n>
      Semantic search by vector similarity.
      Optional: [yellow]USING MODEL[/yellow] '<model>'
      Optional: [yellow]USING HYBRID[/yellow] [FUSION 'rrf|dbsf'] [DENSE MODEL '<model>'] [SPARSE MODEL '<model>']
      Optional: [yellow]USING SPARSE[/yellow] [MODEL '<model>']   sparse-vector-only search
      Optional: [yellow]WHERE[/yellow] <filter>   (e.g. WHERE year > 2020 AND status = 'ok')
      Optional: [yellow]RERANK[/yellow] [MODEL '<model>']   rerank results with a cross-encoder
      Optional: [yellow]EXACT[/yellow]   bypass HNSW and perform exact search
      Optional: [yellow]WITH[/yellow] { hnsw_ef: <int>, exact: <bool>, acorn: <bool> }   search parameters

  [yellow]RECOMMEND FROM[/yellow] <name> [yellow]POSITIVE IDS[/yellow] (<id>, ...)
      Find points similar to known examples.
      Optional: [yellow]NEGATIVE IDS[/yellow] (<id>, ...)
      Optional: [yellow]STRATEGY[/yellow] 'average_vector|best_score|sum_scores'
      Optional: [yellow]WHERE[/yellow] <filter>
      Requires: [yellow]LIMIT[/yellow] <n>

  [yellow]DELETE FROM[/yellow] <name> [yellow]WHERE id =[/yellow] '<id>'
      Delete a point by its ID.

Script files (in-shell):
  [yellow]EXECUTE[/yellow] <path>   or   [yellow]\\e[/yellow] <path>
      Run a .qql script file. Statements are executed in order.
      Lines starting with [yellow]--[/yellow] are treated as comments and ignored.

  [yellow]DUMP[/yellow] <name> <output.qql>   or   [yellow]DUMP COLLECTION[/yellow] <name> <output.qql>
      Export all points in a collection to a .qql script file.
      The file can be re-imported with EXECUTE.

Keyboard shortcuts:
  Left/Right arrows   move cursor within the current line
  Up/Down arrows   scroll through command history
  Ctrl-A / Ctrl-E   jump to beginning / end of line
  Ctrl-C   cancel current input
  Ctrl-D   exit shell

Type [bold]exit[/bold] or [bold]quit[/bold] to leave the shell.
"""


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """QQL — Qdrant Query Language CLI."""
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand is None:
        # No subcommand → load saved config and launch REPL
        cfg = load_config()
        if cfg is None:
            err_console.print(
                "[bold red]Not connected.[/bold red] "
                "Run: [bold]qql connect --url <url>[/bold]"
            )
            sys.exit(1)
        _launch_repl(cfg)


# ── connect ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--url", required=True, help="Qdrant instance URL, e.g. http://localhost:6333")
@click.option("--secret", default=None, help="API key / secret (optional)")
def connect(url: str, secret: str | None) -> None:
    """Connect to a Qdrant instance and launch the QQL shell."""
    from qdrant_client import QdrantClient

    console.print(f"Connecting to [bold]{url}[/bold]...")

    try:
        client = QdrantClient(url=url, api_key=secret)
        client.get_collections()  # validate connection
    except Exception as e:
        err_console.print(f"[bold red]Connection failed:[/bold red] {e}")
        sys.exit(1)

    cfg = QQLConfig(url=url, secret=secret)
    save_config(cfg)
    console.print(f"[bold green]Connected.[/bold green] Config saved to ~/.qql/config.json\n")
    _launch_repl(cfg)


# ── disconnect ─────────────────────────────────────────────────────────────────

@main.command()
def disconnect() -> None:
    """Remove saved connection config."""
    delete_config()
    console.print("Disconnected. Config removed.")


# ── execute ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True, readable=True))
@click.option(
    "--stop-on-error",
    is_flag=True,
    default=False,
    help="Halt execution on the first statement error (default: continue all).",
)
def execute(file: str, stop_on_error: bool) -> None:
    """Execute a .qql script file against the connected Qdrant instance.

    Lines beginning with -- are treated as comments and skipped.
    Each QQL statement is executed in order and its result is printed.
    """
    from qdrant_client import QdrantClient

    cfg = load_config()
    if cfg is None:
        err_console.print(
            "[bold red]Not connected.[/bold red] "
            "Run: [bold]qql connect --url <url>[/bold]"
        )
        sys.exit(1)

    try:
        client = QdrantClient(url=cfg.url, api_key=cfg.secret)
        client.get_collections()
    except Exception as e:
        err_console.print(f"[bold red]Connection failed:[/bold red] {e}")
        sys.exit(1)

    from .executor import Executor
    from .script import run_script

    executor = Executor(client, cfg)
    console.print(f"[bold cyan]Executing:[/bold cyan] {file}\n")

    ok, fail = run_script(file, executor, console, err_console, stop_on_error)
    total = ok + fail

    if fail == 0:
        console.print(
            f"\n[bold green]Done.[/bold green] "
            f"{total}/{total} statement(s) succeeded."
        )
    else:
        console.print(
            f"\n[bold yellow]Done.[/bold yellow] "
            f"{ok}/{total} succeeded, [bold red]{fail} failed[/bold red]."
        )
        sys.exit(1)


# ── dump ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("collection")
@click.argument("output", type=click.Path())
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Points per INSERT BULK batch in the generated script.",
)
def dump(collection: str, output: str, batch_size: int) -> None:
    """Dump a collection to a .qql script file.

    OUTPUT is the path for the generated .qql file.
    The file contains CREATE COLLECTION + INSERT BULK statements and can be
    re-imported with: qql execute <output>
    """
    from qdrant_client import QdrantClient

    cfg = load_config()
    if cfg is None:
        err_console.print(
            "[bold red]Not connected.[/bold red] "
            "Run: [bold]qql connect --url <url>[/bold]"
        )
        sys.exit(1)

    try:
        client = QdrantClient(url=cfg.url, api_key=cfg.secret)
        client.get_collections()
    except Exception as e:
        err_console.print(f"[bold red]Connection failed:[/bold red] {e}")
        sys.exit(1)

    from .dumper import dump_collection

    console.print(
        f"[bold cyan]Dumping:[/bold cyan] '{collection}'  ->  {output}\n"
    )
    written, skipped = dump_collection(
        collection, output, client, console, err_console, batch_size=batch_size
    )

    if written == 0 and skipped == 0:
        # collection not found — error already printed by dump_collection
        sys.exit(1)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{written} point(s) written"
        + (f", [yellow]{skipped} skipped[/yellow] (no 'text' field)" if skipped else "")
        + f"."
    )


# ── REPL ───────────────────────────────────────────────────────────────────────

def _launch_repl(cfg: QQLConfig) -> None:
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=cfg.url, api_key=cfg.secret)
        client.get_collections()
    except Exception as e:
        err_console.print(f"[bold red]Could not connect to {cfg.url}:[/bold red] {e}")
        err_console.print("Run [bold]qql connect --url <url>[/bold] to update your connection.")
        sys.exit(1)

    executor = Executor(client, cfg)

    console.print(f"[bold cyan]QQL Interactive Shell[/bold cyan]  •  {cfg.url}")
    console.print("Type [bold]help[/bold] for available commands or [bold]exit[/bold] to quit.\n")

    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    while True:
        try:
            query = session.prompt(HTML("<ansigreen><b>qql&gt;</b></ansigreen> ")).strip()
        except KeyboardInterrupt:
            # Ctrl-C clears the current line; continue the loop
            continue
        except EOFError:
            # Ctrl-D exits
            console.print("\nBye.")
            break

        if not query:
            continue

        low = query.lower()
        if low in ("exit", "quit", "\\q", ":q"):
            console.print("Bye.")
            break
        if low in ("help", "\\h", "?"):
            console.print(HELP_TEXT)
            continue

        # ── EXECUTE <path> / \e <path> — run a .qql script file ──────────
        if low.startswith("execute ") or low.startswith("\\e "):
            script_path = query.split(None, 1)[1].strip()
            from .script import run_script
            ok, fail = run_script(script_path, executor, console, err_console)
            total = ok + fail
            if fail == 0:
                console.print(
                    f"[bold green]Done.[/bold green] "
                    f"{total}/{total} statement(s) succeeded."
                )
            else:
                console.print(
                    f"[bold yellow]Done.[/bold yellow] "
                    f"{ok}/{total} succeeded, [bold red]{fail} failed[/bold red]."
                )
            continue

        # ── DUMP [COLLECTION] <name> <file> — export collection to .qql ──
        # Accepts both:
        #   DUMP COLLECTION <name> <output.qql>
        #   DUMP <name> <output.qql>
        if low.startswith("dump "):
            parts = query.split(None, 3)  # up to 4 tokens
            if len(parts) >= 2 and parts[1].lower() == "collection":
                # DUMP COLLECTION <name> <file>
                if len(parts) < 4:
                    err_console.print(
                        "[bold red]Usage:[/bold red] DUMP COLLECTION <name> <output.qql>"
                    )
                    continue
                coll_name, out_path = parts[2], parts[3]
            else:
                # DUMP <name> <file>
                if len(parts) < 3:
                    err_console.print(
                        "[bold red]Usage:[/bold red] DUMP <name> <output.qql>"
                    )
                    continue
                coll_name, out_path = parts[1], parts[2]
            from .dumper import dump_collection
            console.print(
                f"[bold cyan]Dumping:[/bold cyan] '{coll_name}'  ->  {out_path}\n"
            )
            written, skipped = dump_collection(
                coll_name, out_path, client, console, err_console
            )
            if written > 0 or skipped == 0:
                console.print(
                    f"[bold green]Done.[/bold green] "
                    f"{written} point(s) written"
                    + (f", [yellow]{skipped} skipped[/yellow] (no 'text' field)" if skipped else "")
                    + "."
                )
            continue

        _run_and_print(executor, query)


def _format_collection_diagnostics(data: dict) -> str:
    """Format SHOW COLLECTION <name> diagnostics into a rich string."""
    lines = []

    lines.append(f"[bold cyan]Collection:[/bold cyan] {data['name']}")
    lines.append(f"  Status               : {data['status']}")
    lines.append(f"  Points               : {data['points_count']}")
    lines.append(f"  Indexed vectors      : {data['indexed_vectors_count']}")
    lines.append(f"  Segments             : {data['segments_count']}")
    lines.append(f"  Topology             : {data['topology']}")

    # Vectors
    vectors = data["vectors"]
    for vname, vconf in vectors.items():
        label = f"  Vector '{vname}'" if vname else "  Vector"
        lines.append(f"{label}        : {vconf['size']} dims, {vconf['distance']} distance")

    # Sparse vectors
    if data["sparse_vectors"]:
        for sname, sconf in data["sparse_vectors"].items():
            lines.append(f"  Sparse '{sname}'      : modifier={sconf['modifier']}")

    lines.append(f"  Quantization         : {data['quantization'] or 'none'}")

    # HNSW config
    hnsw = data["hnsw_config"]
    lines.append(f"  HNSW M               : {hnsw['m']}")
    lines.append(f"  HNSW ef_construct    : {hnsw['ef_construct']}")
    if hnsw.get("full_scan_threshold") is not None:
        lines.append(f"  HNSW full_scan_thres : {hnsw['full_scan_threshold']}")
    if hnsw.get("max_indexing_threads") is not None:
        lines.append(f"  HNSW max_idx_threads : {hnsw['max_indexing_threads']}")
    if hnsw.get("on_disk") is not None:
        lines.append(f"  HNSW on_disk         : {hnsw['on_disk']}")
    if hnsw.get("payload_m") is not None:
        lines.append(f"  HNSW payload_m       : {hnsw['payload_m']}")

    # Payload schema
    schema = data["payload_schema"]
    if schema:
        lines.append("  Payload indexes:")
        for field, dtype in schema.items():
            lines.append(f"    {field}: {dtype}")
    else:
        lines.append("  Payload indexes      : none")

    # Sharding
    sh = data["sharding"]
    lines.append(f"  Shards               : {sh['shard_number']}")
    lines.append(f"  Replicas              : {sh['replication_factor']}")
    lines.append(f"  Write consistency    : {sh['write_consistency_factor']}")

    return "\n".join(lines)


def _run_and_print(executor: Executor, query: str) -> None:
    try:
        tokens = Lexer().tokenize(query)
        node = Parser(tokens).parse()
        result = executor.execute(node)
    except QQLError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        return
    except Exception as e:
        err_console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        return

    if not result.success:
        err_console.print(f"[bold red]Failed:[/bold red] {result.message}")
        return

    console.print(f"[bold green]OK[/bold green] {result.message}")

    if result.data is None:
        return

    # Pretty-print collections list
    if isinstance(result.data, list) and all(isinstance(x, str) for x in result.data):
        if result.data:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Collection")
            for name in result.data:
                table.add_row(name)
            console.print(table)
        return

    # Pretty-print SHOW COLLECTION <name> diagnostics
    if isinstance(result.data, dict) and "topology" in result.data:
        console.print(_format_collection_diagnostics(result.data))
        return

    # Pretty-print search results
    if isinstance(result.data, list) and result.data and isinstance(result.data[0], dict) and "score" in result.data[0]:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Score", justify="right")
        table.add_column("ID")
        table.add_column("Payload")
        for hit in result.data:
            table.add_row(
                str(hit["score"]),
                hit["id"],
                str(hit["payload"]),
            )
        console.print(table)
        return

    # Pretty-print scroll results
    if isinstance(result.data, dict) and "points" in result.data and "next_offset" in result.data:
        points = result.data["points"]
        if points:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("ID")
            table.add_column("Payload")
            for point in points:
                table.add_row(point["id"], str(point["payload"]))
            console.print(table)
        if result.data["next_offset"] is not None:
            console.print(f"[dim]next_offset: {result.data['next_offset']}[/dim]")
        return

    # Pretty-print SELECT result
    if isinstance(result.data, dict) and "id" in result.data and "payload" in result.data:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID")
        table.add_column("Payload")
        table.add_row(str(result.data["id"]), str(result.data["payload"]))
        console.print(table)
        return

    # Fallback: print data as-is
    console.print(result.data)
