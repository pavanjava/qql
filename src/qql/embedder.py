from __future__ import annotations


def _load_text_cross_encoder_class():
    """Load the fastembed cross-encoder class across package layouts."""
    try:
        from fastembed import TextCrossEncoder

        return TextCrossEncoder
    except ImportError:
        from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

        return TextCrossEncoder


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


class SparseEmbedder:
    """Sparse BM25-style embedder using fastembed.SparseTextEmbedding.

    Returns dicts with "indices" and "values" lists (not numpy arrays),
    ready for direct construction of qdrant_client SparseVector objects.

    Uses asymmetric embedding: embed() for document indexing, query_embed()
    for query-time encoding (BM25 IDF weighting differs at query vs. index time).
    """

    DEFAULT_MODEL = "Qdrant/bm25"

    # Class-level cache mirrors Embedder's pattern
    _cache: dict[str, object] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        if model_name not in SparseEmbedder._cache:
            from fastembed import SparseTextEmbedding

            SparseEmbedder._cache[model_name] = SparseTextEmbedding(model_name)
        self._model = SparseEmbedder._cache[model_name]

    def embed(self, text: str) -> dict[str, list]:
        """Embed a document string. Returns {"indices": [...], "values": [...]}."""
        result = next(iter(self._model.embed([text])))  # type: ignore[attr-defined]
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}

    def query_embed(self, text: str) -> dict[str, list]:
        """Embed a query string (BM25 applies different IDF weighting at query time)."""
        result = next(iter(self._model.query_embed(text)))  # type: ignore[attr-defined]
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}


class CrossEncoderEmbedder:
    """Cross-encoder reranker using fastembed.TextCrossEncoder.

    Jointly encodes (query, document) pairs to produce relevance scores.
    Higher score = more relevant. No new package dependencies —
    TextCrossEncoder is included in the fastembed package bundled with
    qdrant-client[fastembed].
    """

    DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Class-level cache mirrors Embedder's pattern
    _cache: dict[str, object] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        if model_name not in CrossEncoderEmbedder._cache:
            text_cross_encoder_cls = _load_text_cross_encoder_class()
            CrossEncoderEmbedder._cache[model_name] = text_cross_encoder_cls(model_name)
        self._model = CrossEncoderEmbedder._cache[model_name]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return a relevance score for each (query, document) pair.

        Scores are raw logits — higher means more relevant.
        The returned list is the same length as ``documents`` and in the same order.
        """
        return list(self._model.rerank(query, documents))  # type: ignore[attr-defined]
