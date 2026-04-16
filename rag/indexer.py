"""
rag/indexer.py — File reading and chunking for KOBRA's knowledge base.

Supports: .py .md .txt .pdf .docx .js .ts .jsx .json .yaml .html .ipynb .rst .toml .csv
Chunks text into overlapping windows for ChromaDB storage.
"""

import json
import logging
import os
import re

import config

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".py", ".md", ".txt", ".pdf", ".docx", ".js", ".ts", ".jsx",
    ".json", ".yaml", ".yml", ".html", ".htm", ".ipynb", ".rst",
    ".toml", ".csv",
}

_MAX_FILE_BYTES = config.RAG_MAX_FILE_SIZE_MB * 1024 * 1024


class FileIndexer:
    def extract_text(self, filepath: str) -> str | None:
        """
        Extract text from a file. Returns None if unsupported, too large, or failed.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return None

        try:
            if os.path.getsize(filepath) > _MAX_FILE_BYTES:
                logger.debug("Skipping large file: %s", filepath)
                return None
        except OSError:
            return None

        try:
            if ext == ".pdf":
                return self._extract_pdf(filepath)
            elif ext == ".docx":
                return self._extract_docx(filepath)
            elif ext == ".ipynb":
                return self._extract_ipynb(filepath)
            elif ext in (".html", ".htm"):
                return self._extract_html(filepath)
            else:
                return self._extract_text(filepath)
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", filepath, exc)
            return None

    # ── Extractors ────────────────────────────────────────────────────────────

    def _extract_pdf(self, path: str) -> str:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)

    def _extract_docx(self, path: str) -> str:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_ipynb(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if source.strip():
                cells.append(f"[{cell['cell_type']}]\n{source}")
        return "\n\n".join(cells)

    def _extract_html(self, path: str) -> str:
        from bs4 import BeautifulSoup
        with open(path, encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")

    def _extract_text(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, filepath: str) -> list[dict]:
        """
        Split text into overlapping chunks for embedding.
        Each chunk: {text, source, filename, chunk_index, file_modified}
        """
        if not text or not text.strip():
            return []

        file_modified = str(int(os.path.getmtime(filepath)))
        filename = os.path.basename(filepath)

        # Split on double newlines first (paragraph/function boundaries)
        raw_chunks = re.split(r'\n{2,}', text)
        # Then hard-limit each chunk to ~RAG_CHUNK_SIZE words
        words_per_chunk = config.RAG_CHUNK_SIZE
        overlap = config.RAG_CHUNK_OVERLAP

        # Flatten into word tokens with paragraph markers
        all_words: list[str] = []
        for para in raw_chunks:
            words = para.split()
            if words:
                all_words.extend(words)
                all_words.append("\n\n")  # paragraph boundary marker

        chunks: list[dict] = []
        i = 0
        chunk_idx = 0
        while i < len(all_words):
            window = all_words[i: i + words_per_chunk]
            chunk_text = " ".join(w for w in window if w != "\n\n").strip()
            # Restore paragraph breaks
            chunk_text = re.sub(r'\s+', ' ', chunk_text)

            if len(chunk_text) >= 20:  # skip noise
                chunks.append({
                    "text":          chunk_text,
                    "source":        filepath,
                    "filename":      filename,
                    "chunk_index":   chunk_idx,
                    "file_modified": file_modified,
                })
                chunk_idx += 1

            i += words_per_chunk - overlap

        return chunks
