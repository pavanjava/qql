from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".qql"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "sparse"


@dataclass
class QQLConfig:
    url: str
    secret: str | None = None
    default_model: str = DEFAULT_MODEL
    default_dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME
    default_sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME
    verify: bool | str = True


def save_config(cfg: QQLConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config() -> QQLConfig | None:
    """Return saved config, or None if not yet connected."""
    if not CONFIG_PATH.exists():
        return None
    with CONFIG_PATH.open() as f:
        data = json.load(f)
    return QQLConfig(
        url=data["url"],
        secret=data.get("secret"),
        default_model=data.get("default_model", DEFAULT_MODEL),
        default_dense_vector_name=data.get(
            "default_dense_vector_name", DEFAULT_DENSE_VECTOR_NAME
        ),
        default_sparse_vector_name=data.get(
            "default_sparse_vector_name", DEFAULT_SPARSE_VECTOR_NAME
        ),
        verify=data.get("verify", True),
    )


def delete_config() -> None:
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
