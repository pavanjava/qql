class QQLError(Exception):
    """Base for all QQL errors."""


class QQLSyntaxError(QQLError):
    """Raised by Lexer or Parser for malformed input."""

    def __init__(self, message: str, pos: int | None = None) -> None:
        self.pos = pos
        suffix = f" (at position {pos})" if pos is not None else ""
        super().__init__(f"{message}{suffix}")


class QQLRuntimeError(QQLError):
    """Raised by Executor when the operation fails."""
