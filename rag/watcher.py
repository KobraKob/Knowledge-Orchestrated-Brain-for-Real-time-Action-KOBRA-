"""
rag/watcher.py — Watchdog-based file watcher for KOBRA's knowledge base.

Monitors watched folders for file changes and re-indexes automatically.
Debounces rapid saves (e.g. mid-save writes) with a 3-second delay per file.
"""

import logging
import os
import threading
import time

import config

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".py", ".md", ".txt", ".pdf", ".docx", ".js", ".ts", ".jsx",
    ".json", ".yaml", ".yml", ".html", ".htm", ".ipynb", ".rst",
    ".toml", ".csv",
}
_DEBOUNCE_SECONDS = 3.0
_MAX_FILE_BYTES = config.RAG_MAX_FILE_SIZE_MB * 1024 * 1024


class FolderWatcher:
    def __init__(self, indexer, embedder, store) -> None:
        self._indexer = indexer
        self._embedder = embedder
        self._store = store
        self._observer = None
        self._pending: dict[str, float] = {}  # path → scheduled process time
        self._lock = threading.Lock()

    # ── Initial full index ────────────────────────────────────────────────────

    def index_all(self, folders: list[str]) -> None:
        """Index all files in watched folders. Skips files unchanged since last index."""
        total_chunks = 0
        total_files = 0

        for folder in folders:
            if not os.path.isdir(folder):
                logger.debug("[RAG] Folder not found, skipping: %s", folder)
                continue
            for dirpath, _dirnames, filenames in os.walk(folder):
                # Skip hidden directories
                _dirnames[:] = [d for d in _dirnames if not d.startswith(".")]
                for fname in filenames:
                    if fname.startswith("."):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in _SUPPORTED_EXTENSIONS:
                        continue
                    filepath = os.path.join(dirpath, fname)
                    try:
                        if os.path.getsize(filepath) > _MAX_FILE_BYTES:
                            continue
                        mtime = str(int(os.path.getmtime(filepath)))
                        indexed_mtime = self._store.get_indexed_modified(filepath)
                        if indexed_mtime == mtime:
                            continue  # unchanged
                        chunks = self._index_file(filepath)
                        total_chunks += len(chunks)
                        if chunks:
                            total_files += 1
                    except Exception as exc:
                        logger.debug("Skipping %s: %s", filepath, exc)

        logger.info("[RAG] Initial index: %d chunks from %d files", total_chunks, total_files)

    def _index_file(self, filepath: str) -> list[dict]:
        """Extract, chunk, embed, and upsert one file. Returns the chunks."""
        text = self._indexer.extract_text(filepath)
        if not text:
            return []
        chunks = self._indexer.chunk_text(text, filepath)
        if not chunks:
            return []
        embeddings = self._embedder.embed_batch([c["text"] for c in chunks])
        self._store.upsert(chunks, embeddings)
        logger.debug("[RAG] Indexed %d chunks from %s", len(chunks), filepath)
        return chunks

    # ── Live watching ─────────────────────────────────────────────────────────

    def start(self, folders: list[str]) -> None:
        """Start the watchdog observer on all folders. Runs as a daemon thread."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.warning("[RAG] watchdog not installed — live file watching disabled.")
            return

        class _Handler(FileSystemEventHandler):
            def __init__(self_, watcher):
                self_._watcher = watcher

            def on_any_event(self_, event):
                if event.is_directory:
                    return
                path = getattr(event, "dest_path", None) or event.src_path
                self_._watcher._schedule(path, event.event_type)

        observer = Observer()
        handler = _Handler(self)
        for folder in folders:
            if os.path.isdir(folder):
                observer.schedule(handler, folder, recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer

        # Background thread processes debounced events
        t = threading.Thread(target=self._process_loop, daemon=True, name="rag-watcher")
        t.start()

        logger.info("[RAG] Watching %d folder(s)", len([f for f in folders if os.path.isdir(f)]))

    def _schedule(self, path: str, event_type: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS or os.path.basename(path).startswith("."):
            return
        with self._lock:
            if event_type == "deleted":
                self._pending[path] = -1.0  # sentinel for deletion
            else:
                self._pending[path] = time.time() + _DEBOUNCE_SECONDS

    def _process_loop(self) -> None:
        while True:
            time.sleep(1.0)
            now = time.time()
            with self._lock:
                due = {p: t for p, t in self._pending.items()
                       if t == -1.0 or now >= t}
                for p in due:
                    del self._pending[p]

            for path, trigger_time in due.items():
                try:
                    if trigger_time == -1.0:
                        self._store.delete_file(path)
                        logger.info("[RAG] Removed: %s", path)
                    else:
                        self._index_file(path)
                        logger.info("[RAG] Re-indexed: %s", path)
                except Exception as exc:
                    logger.warning("[RAG] Event handler failed for %s: %s", path, exc)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
