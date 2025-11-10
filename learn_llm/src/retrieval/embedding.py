"""
Embedding utilities that hide provider differences and expose a single API.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Iterable, List, Optional, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from sklearn.feature_extraction.text import TfidfVectorizer

Vector = List[float]


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot fulfil the request."""


@dataclass
class EmbeddingConfig:
    """
    Generic configuration shared across all embedders.

    provider:
        "openai" (default) uses any OpenAI-compatible endpoint.
        "local" uses a TF-IDF fallback so you can prototype without an API key.
    """

    provider: str = "local"
    model: str = "text-embedding-3-small"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 32
    normalize: bool = True
    max_retries: int = 3


class BaseEmbedder:
    """Interface all concrete embedders implement."""

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> Vector:
        # Default implementation keeps provider code small.
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


class OpenAIEmbedder(BaseEmbedder):
    """Wraps any OpenAI-compatible embedding endpoint."""

    def __init__(self, config: EmbeddingConfig):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIEmbedder")
        api_key = config.api_key or os.getenv(config.api_key_env)
        if not api_key:
            raise EmbeddingError(
                f"Missing API key for provider; set {config.api_key_env} or pass api_key"
            )
        self._config = config
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        if not texts:
            return []
        vectors: List[Vector] = []
        for batch in _batched(texts, self._config.batch_size):
            response = self._client.embeddings.create(
                model=self._config.model,
                input=batch,
            )
            for item in response.data:
                vectors.append(self._post_process(item.embedding))
        return vectors

    def _post_process(self, vector: Sequence[float]) -> Vector:
        data = list(vector)
        return _normalize(data) if self._config.normalize else data


class LocalTfidfEmbedder(BaseEmbedder):
    """
    Lightweight fallback embedder powered by TF-IDF. Keeps the rest of the
    retrieval stack testable when external APIs are unavailable.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        max_features: int = 768,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self._config = config
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self._fitted = False

    def fit_corpus(self, corpus: Iterable[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        if not texts:
            return []
        if not self._fitted:
            matrix = self._vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray()
        return [_normalize(vec.tolist()) if self._config.normalize else vec.tolist() for vec in dense]

    def embed_query(self, text: str) -> Vector:
        if not self._fitted:
            raise EmbeddingError(
                "Vectorizer not fitted yet. Call fit_corpus or embed_documents first."
            )
        vector = self._vectorizer.transform([text]).toarray()[0].tolist()
        return _normalize(vector) if self._config.normalize else vector


def build_embedder(config: Optional[EmbeddingConfig] = None) -> BaseEmbedder:
    """
    Factory that returns the right embedder for the provided configuration.
    """

    config = config or EmbeddingConfig()
    provider = config.provider.lower()
    if provider in {"openai", "deepseek", "azure"}:
        return OpenAIEmbedder(config)
    if provider == "local":
        return LocalTfidfEmbedder(config)
    raise EmbeddingError(f"Unknown embedding provider: {config.provider}")


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    size = max(batch_size, 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


if __name__ == "__main__":  # rudimentary smoke test
    logging.basicConfig(level=logging.INFO)
    cfg = EmbeddingConfig(provider="local")
    embedder = build_embedder(cfg)
    sample_docs = [
        "Cough with fever suggests an upper respiratory infection.",
        "Chest pain plus hypertension history requires cardiac screening.",
    ]
    vectors = embedder.embed_documents(sample_docs)
    logging.info("Generated %d vectors with dim=%d", len(vectors), len(vectors[0]))
