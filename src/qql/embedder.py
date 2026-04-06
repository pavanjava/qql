from __future__ import annotations


class Embedder:
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Class-level cache: model_name → TextEmbedding instance
    # Avoids reloading the same model across multiple Embedder() calls in one session.
    _cache: dict[str, object] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        if model_name not in Embedder._cache:
            # Import here so the module loads even without fastembed installed
            # (the error surfaces only when embedding is actually attempted)
            from fastembed import TextEmbedding

            Embedder._cache[model_name] = TextEmbedding(model_name)
        self._model = Embedder._cache[model_name]

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return a plain list[float]."""
        result = next(iter(self._model.embed([text])))  # type: ignore[attr-defined]
        return result.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings."""
        return [v.tolist() for v in self._model.embed(texts)]  # type: ignore[attr-defined]

    @property
    def dimensions(self) -> int:
        """Return the vector dimensionality by embedding a dummy string."""
        return len(self.embed("probe"))
