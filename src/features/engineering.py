"""RAG retrieval pipeline for the Internal Copilot.

Manages the FAISS vector index over internal policy documents,
handles chunk retrieval, and applies context grounding scoring
before passing context to the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small
TOP_K = 5
GROUNDING_THRESHOLD = 0.70


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk."""

    text: str
    source: str
    score: float


@dataclass
class RAGResult:
    """Result of a RAG retrieval for a single query."""

    query: str
    chunks: list[RetrievedChunk]
    grounding_score: float
    is_grounded: bool


class PolicyVectorIndex:
    """FAISS-backed vector index over internal policy documents."""

    def __init__(self, index_path: str, metadata_path: str) -> None:
        import faiss  # noqa: PLC0415
        import json  # noqa: PLC0415

        self._index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self._metadata: list[dict] = json.load(f)
        logger.info(
            "Loaded policy index: %d chunks from %s",
            self._index.ntotal,
            index_path,
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> RAGResult:
        """Retrieve the top-K most relevant policy chunks.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query vector of shape (EMBEDDING_DIM,).
        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        RAGResult with retrieved chunks and grounding score.
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(query, top_k)

        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            chunks.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta["source"],
                    score=float(dist),
                )
            )

        grounding_score = float(np.mean([c.score for c in chunks])) if chunks else 0.0
        return RAGResult(
            query="",
            chunks=chunks,
            grounding_score=grounding_score,
            is_grounded=grounding_score >= GROUNDING_THRESHOLD,
        )
