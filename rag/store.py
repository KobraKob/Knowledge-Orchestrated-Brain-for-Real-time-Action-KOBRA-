"""
rag/store.py — ChromaDB vector store for KOBRA's knowledge base.

Persists embeddings to kobra_rag_db/ on disk.
Handles upsert (idempotent re-indexing), delete by file, and cosine similarity query.
"""

import logging

import config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=config.RAG_DB_PATH)
            self._collection = self._client.get_or_create_collection(
                name="kobra_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[RAG] VectorStore ready at %s (%d chunks)",
                        config.RAG_DB_PATH, self._collection.count())
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Insert or update chunks. Uses source::chunk_N as stable ID."""
        if not chunks:
            return
        # Filter out empty embeddings
        valid = [(c, e) for c, e in zip(chunks, embeddings) if e]
        if not valid:
            return
        chunks_v, embeddings_v = zip(*valid)

        ids = [f"{c['source']}::chunk_{c['chunk_index']}" for c in chunks_v]
        documents = [c["text"] for c in chunks_v]
        metadatas = [{
            "source":        c["source"],
            "filename":      c["filename"],
            "chunk_index":   c["chunk_index"],
            "file_modified": c["file_modified"],
        } for c in chunks_v]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=list(embeddings_v),
            metadatas=metadatas,
        )

    def delete_file(self, filepath: str) -> None:
        """Delete all chunks belonging to a file."""
        try:
            self._collection.delete(where={"source": filepath})
            logger.info("[RAG] Deleted chunks for: %s", filepath)
        except Exception as exc:
            logger.warning("delete_file failed for %s: %s", filepath, exc)

    def query(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        """
        Return top N semantically similar chunks.
        Each result: {text, source, filename, distance}
        """
        if not embedding:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text":     doc,
                    "source":   meta.get("source", ""),
                    "filename": meta.get("filename", ""),
                    "distance": dist,
                })
            return chunks
        except Exception as exc:
            logger.error("VectorStore query failed: %s", exc)
            return []

    def get_indexed_modified(self, filepath: str) -> str | None:
        """
        Return the file_modified timestamp for the first indexed chunk of a file,
        or None if the file isn't indexed.
        """
        try:
            results = self._collection.get(
                where={"source": filepath},
                limit=1,
                include=["metadatas"],
            )
            if results["metadatas"]:
                return results["metadatas"][0].get("file_modified")
        except Exception:
            pass
        return None

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return unique file paths currently indexed."""
        try:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            return sorted({m["source"] for m in all_meta if "source" in m})
        except Exception:
            return []
