"""
Lightweight vector store abstractions + an in-memory reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


Vector = Sequence[float]
Metadata = Dict[str, Any]
FilterFn = Callable[[Metadata], bool]


@dataclass
class Document:
    """Chunk of text + embedding."""

    doc_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    embedding: Optional[Vector] = None


@dataclass
class SearchResult:
    document: Document
    score: float


class VectorStore:
    """Interface every vector store should expose."""

    def add(self, documents: Iterable[Document]) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[FilterFn] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError

    def persist(self, path: str) -> None:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """
    Minimal cosine-similarity store.
    Works for quick experiments before plugging in FAISS/Chroma.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._vectors: Optional[np.ndarray] = None
        self._documents: List[Document] = []

    def add(self, documents: Iterable[Document]) -> None:
        embeddings = []
        docs: List[Document] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} missing embedding")
            vector = np.asarray(doc.embedding, dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError("Embedding must be 1-D")
            if self.normalize:
                vector = _l2_normalize(vector)
            embeddings.append(vector)
            docs.append(doc)

        if not embeddings:
            return

        new_vectors = np.vstack(embeddings)
        if self._vectors is None:
            self._vectors = new_vectors
            self._documents = docs
        else:
            if new_vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("Embedding dimension mismatch")
            self._vectors = np.vstack([self._vectors, new_vectors])
            self._documents.extend(docs)

    def search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[FilterFn] = None,
    ) -> List[SearchResult]:
        if self._vectors is None or not len(self._documents):
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("Query embedding must be 1-D")
        if self.normalize:
            query = _l2_normalize(query)
        scores = self._vectors @ query
        ranked_indices = np.argsort(scores)[::-1]

        matches: List[SearchResult] = []
        for idx in ranked_indices:
            doc = self._documents[idx]
            if metadata_filter and not metadata_filter(doc.metadata):
                continue
            matches.append(SearchResult(document=doc, score=float(scores[idx])))
            if len(matches) >= top_k:
                break
        return matches

    def persist(self, path: str) -> None:
        if self._vectors is None:
            raise ValueError("Nothing to persist; store is empty")
        target = Path(path)
        payload = [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in self._documents
        ]
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(target.with_suffix(".npz"), vectors=self._vectors)

    @classmethod
    def load(cls, path: str) -> "InMemoryVectorStore":
        meta_path = Path(path)
        vector_path = meta_path.with_suffix(".npz")
        documents: List[Document] = []
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            for item in data:
                documents.append(
                    Document(
                        doc_id=item["doc_id"],
                        text=item["text"],
                        metadata=item.get("metadata", {}),
                    )
                )
        store = cls()
        if vector_path.exists():
            vectors = np.load(vector_path)["vectors"]
            store._vectors = vectors
            for doc, vector in zip(documents, vectors):
                doc.embedding = vector.tolist()
            store._documents = documents
        return store


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm == 0:
        return vector
    return vector / norm


if __name__ == "__main__":
    docs = [
        Document("1", "上呼吸道感染治疗指南", {"topic": "infection"}, [0.1, 0.3, 0.5]),
        Document("2", "高血压随访记录", {"topic": "cardio"}, [0.2, 0.1, 0.7]),
    ]
    store = InMemoryVectorStore()
    store.add(docs)
    query_vec = [0.1, 0.2, 0.6]
    results = store.search(query_vec, top_k=1)
    for res in results:
        print(res.document.doc_id, res.score)
