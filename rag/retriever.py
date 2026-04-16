"""
rag/retriever.py — Query the knowledge base and return formatted context.

Two public methods:
  query(question)              → formatted context string
  query_with_sources(question) → (context_string, [source_file_paths])
"""

import logging

import config

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder, store) -> None:
        self._embedder = embedder
        self._store = store

    def query(self, question: str, n_results: int | None = None) -> str:
        context, _ = self.query_with_sources(question, n_results)
        return context

    def query_with_sources(
        self,
        question: str,
        n_results: int | None = None,
    ) -> tuple[str, list[str]]:
        """
        Embed the question, retrieve top K chunks, rerank, and format.
        Returns (context_string, list_of_source_files).
        """
        k = n_results or config.RAG_TOP_K
        try:
            embedding = self._embedder.embed(question)
        except Exception as exc:
            logger.error("[RAG] Embedding failed: %s", exc)
            return "", []

        chunks = self._store.query(embedding, n_results=k)
        if not chunks:
            return "", []

        # Rerank: boost chunks containing question keywords
        question_keywords = set(question.lower().split())
        for chunk in chunks:
            keyword_hits = sum(
                1 for kw in question_keywords
                if kw in chunk["text"].lower() and len(kw) > 3
            )
            # Lower distance = more similar. Subtract keyword bonus.
            chunk["score"] = chunk["distance"] - (keyword_hits * 0.05)

        chunks.sort(key=lambda c: c["score"])

        # Format context block
        parts = []
        sources: list[str] = []
        seen_sources: set[str] = set()

        for chunk in chunks:
            source = chunk["source"]
            filename = chunk["filename"]
            parts.append(f"[Source: {source}]\n{chunk['text']}")
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)

        context = "\n\n".join(parts)
        logger.debug("[RAG] Retrieved %d chunks from %d sources", len(chunks), len(sources))
        return context, sources
