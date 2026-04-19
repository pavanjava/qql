"""QQL script runner — executes .qql files containing multiple statements.

Pipeline:
  1. strip_comments()     — remove -- … to-end-of-line comments
  2. split_statements()   — tokenize once, split on statement-starter
                            keywords at brace/bracket depth 0
  3. run_script()         — parse + execute each chunk, print progress
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .exceptions import QQLError
from .executor import Executor
from .lexer import Lexer, Token, TokenKind
from .parser import Parser

# ── Token sets ────────────────────────────────────────────────────────────────

_STMT_STARTERS = {
    TokenKind.INSERT,
    TokenKind.CREATE,
    TokenKind.DROP,
    TokenKind.SHOW,
    TokenKind.SEARCH,
    TokenKind.DELETE,
}

_DEPTH_OPEN  = {TokenKind.LBRACE, TokenKind.LBRACKET, TokenKind.LPAREN}
_DEPTH_CLOSE = {TokenKind.RBRACE, TokenKind.RBRACKET, TokenKind.RPAREN}

# ── Public helpers ────────────────────────────────────────────────────────────


def strip_comments(text: str) -> str:
    """Remove ``-- ...`` to-end-of-line comments from every line.

    The check is byte-level: ``--`` inside a string literal would also be
    stripped, but that edge case does not occur in practice for QQL scripts.
    """
    lines: list[str] = []
    for line in text.splitlines():
        idx = line.find("--")
        if idx != -1:
            line = line[:idx]
        lines.append(line)
    return "\n".join(lines)


def split_statements(tokens: list[Token]) -> list[list[Token]]:
    """Split a flat token list into per-statement chunks.

    A new chunk begins whenever a statement-starter keyword (INSERT, CREATE,
    DROP, SHOW, SEARCH, DELETE) is encountered at brace/bracket/paren depth 0.
    The EOF sentinel is consumed and never included in any chunk.
    """
    chunks: list[list[Token]] = []
    current: list[Token] = []
    depth = 0

    for tok in tokens:
        if tok.kind == TokenKind.EOF:
            break
        if tok.kind in _DEPTH_OPEN:
            depth += 1
        elif tok.kind in _DEPTH_CLOSE:
            depth -= 1

        # New statement starts when we see a starter at the top level
        if tok.kind in _STMT_STARTERS and depth == 0 and current:
            chunks.append(current)
            current = []

        current.append(tok)

    if current:
        chunks.append(current)

    return chunks


def _stmt_label(chunk: list[Token], max_len: int = 70) -> str:
    """Build a short human-readable label from a statement's token list."""
    parts: list[str] = []
    total = 0
    for tok in chunk:
        word = tok.value if tok.kind != TokenKind.STRING else f"'{tok.value}'"
        if total + len(word) + 1 > max_len:
            parts.append("…")
            break
        parts.append(word)
        total += len(word) + 1
    return " ".join(parts)


# ── Main entry point ──────────────────────────────────────────────────────────


def run_script(
    path: str,
    executor: Executor,
    console: Console,
    err_console: Console,
    stop_on_error: bool = False,
) -> tuple[int, int]:
    """Parse and execute every statement in *path*.

    Returns ``(succeeded, failed)`` counts.
    Prints per-statement progress to *console* / *err_console*.
    If *stop_on_error* is True, halts on the first failure.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        err_console.print(f"[bold red]Cannot read file:[/bold red] {e}")
        return 0, 1

    cleaned = strip_comments(text)
    tokens = Lexer().tokenize(cleaned)
    chunks = split_statements(tokens)

    if not chunks:
        console.print("[yellow]No statements found in script.[/yellow]")
        return 0, 0

    n = len(chunks)
    succeeded = 0
    failed = 0
    eof_tok = Token(TokenKind.EOF, "", 0)

    for i, chunk in enumerate(chunks, 1):
        label = _stmt_label(chunk)
        console.print(f"[dim][[{i}/{n}]][/dim] {label}")

        try:
            node = Parser(chunk + [eof_tok]).parse()
            result = executor.execute(node)
        except QQLError as e:
            err_console.print(f"  [bold red]✗[/bold red] {e}")
            failed += 1
            if stop_on_error:
                break
            continue
        except Exception as e:
            err_console.print(f"  [bold red]✗ Unexpected error:[/bold red] {e}")
            failed += 1
            if stop_on_error:
                break
            continue

        console.print(f"  [bold green]✓[/bold green] {result.message}")
        succeeded += 1

    return succeeded, failed
