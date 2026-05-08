"""
memory/retriever.py — Hybrid BM25 + vector retrieval (Cluster 03: Sovereign Developer OS)

Combines dense vector search (Chroma) with sparse BM25 keyword search
for better recall on code-specific queries.

SDKs: LlamaIndex, Chroma
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity + BM25 keyword search.
    Falls back to vector-only if BM25 is unavailable.
    """

    def __init__(self, rag, config: dict):
        self._rag = rag
        self._top_k = config.get("rag", {}).get("top_k", 8)
        self._alpha = config.get("rag", {}).get("hybrid_alpha", 0.5)  # 0=BM25, 1=vector
        self._bm25_retriever = None
        self._vector_retriever = None

    def build(self) -> None:
        """Build retriever from loaded index."""
        if not self._rag._index:
            logger.warning("Index not loaded — cannot build retriever")
            return

        # Vector retriever
        self._vector_retriever = self._rag._index.as_retriever(
            similarity_top_k=self._top_k
        )

        # BM25 retriever (keyword-based)
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            self._bm25_retriever = BM25Retriever.from_defaults(
                index=self._rag._index,
                similarity_top_k=self._top_k,
            )
            logger.info("Hybrid retriever built (BM25 + vector, alpha=%.2f)", self._alpha)
        except ImportError:
            logger.warning("llama-index-retrievers-bm25 not installed — using vector-only")

    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant nodes for a query.
        Returns list of NodeWithScore objects.
        """
        if self._bm25_retriever and self._vector_retriever:
            return self._hybrid_retrieve(query)
        elif self._vector_retriever:
            return self._vector_retriever.retrieve(query)
        else:
            logger.warning("No retriever available")
            return []

    def _hybrid_retrieve(self, query: str) -> list:
        """Reciprocal Rank Fusion of BM25 + vector results."""
        try:
            from llama_index.core.retrievers import QueryFusionRetriever
            fusion = QueryFusionRetriever(
                retrievers=[self._vector_retriever, self._bm25_retriever],
                similarity_top_k=self._top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=False,
            )
            return fusion.retrieve(query)
        except Exception as e:
            logger.warning("Hybrid fusion failed (%s), falling back to vector", e)
            return self._vector_retriever.retrieve(query)

    def retrieve_as_context(self, query: str) -> str:
        """Retrieve and format nodes as a context string for the LLM."""
        nodes = self.retrieve(query)
        if not nodes:
            return ""
        parts = []
        for i, node in enumerate(nodes):
            file_path = node.metadata.get("file_path", node.metadata.get("file_name", "unknown"))
            parts.append(f"[{i+1}] {file_path}\n{node.get_content()[:600]}")
        return "\n\n---\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "vector_retriever": self._vector_retriever is not None,
            "bm25_retriever": self._bm25_retriever is not None,
            "top_k": self._top_k,
            "alpha": self._alpha,
        }
