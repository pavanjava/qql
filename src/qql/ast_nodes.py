from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InsertStmt:
    collection: str
    values: dict[str, Any]  # must contain "text" key
    model: str | None  # None → use default


@dataclass(frozen=True)
class CreateCollectionStmt:
    collection: str


@dataclass(frozen=True)
class DropCollectionStmt:
    collection: str


@dataclass(frozen=True)
class ShowCollectionsStmt:
    pass


@dataclass(frozen=True)
class SearchStmt:
    collection: str
    query_text: str
    limit: int
    model: str | None


@dataclass(frozen=True)
class DeleteStmt:
    collection: str
    point_id: str | int


# Union type for all statement nodes
ASTNode = (
    InsertStmt
    | CreateCollectionStmt
    | DropCollectionStmt
    | ShowCollectionsStmt
    | SearchStmt
    | DeleteStmt
)
