"""Optional Cognee adapter for KOBRA's existing SQLite memory.

SQLite remains the source of truth. Cognee is an opt-in associative retrieval
sidecar and can be disabled instantly with KOBRA_MEMORY_BACKEND=sqlite.
Cognee operations are best-effort: failures never prevent SQLite writes/reads.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    """Run a coroutine from KOBRA's synchronous memory API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # If called from an event-loop thread, run the coroutine on a short-lived
    # helper thread rather than attempting nested asyncio.run().
    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # surfaced to caller below
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None


class CogneeMemory:
    """Thin, optional wrapper around Cognee's async Python API."""

    def __init__(self) -> None:
        self._cognee: Any | None = None
        self._disabled = False
        self._dataset = config.COGNEE_DATASET

    def _client(self) -> Any | None:
        if self._disabled:
            return None
        if self._cognee is None:
            try:
                self._cognee = importlib.import_module("cognee")
            except ImportError:
                self._disabled = True
                logger.warning(
                    "Cognee memory is enabled but the 'cognee' package is not "
                    "installed. SQLite remains active."
                )
                return None
        return self._cognee

    def remember(self, text: str, *, kind: str, source_id: str) -> bool:
        """Store one memory item in Cognee. Returns whether it succeeded."""
        client = self._client()
        if client is None:
            return False
        try:
            payload = (
                f"Memory type: {kind}\n"
                f"Source ID: {source_id}\n"
                f"Recorded at: {__import__('datetime').datetime.now().isoformat()}\n"
                f"Content:\n{text}"
            )
            _run_async(
                client.remember(
                    data=payload,
                    dataset_name=self._dataset,
                    self_improvement=False,
                )
            )
            return True
        except Exception:
            logger.exception("Cognee write failed; SQLite remains authoritative.")
            return False

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Retrieve associative matches from Cognee without making it mandatory."""
        client = self._client()
        if client is None:
            return []
        try:
            raw = _run_async(
                client.recall(
                    query_text=query,
                    datasets=[self._dataset],
                    top_k=config.COGNEE_TOP_K,
                )
            )
            if raw is None:
                return []
            if isinstance(raw, str):
                return [{"source": "cognee", "content": raw}]
            if isinstance(raw, dict):
                return [{"source": "cognee", **raw}]
            results: list[dict[str, Any]] = []
            for item in raw if isinstance(raw, (list, tuple)) else [raw]:
                if isinstance(item, dict):
                    results.append({"source": "cognee", **item})
                else:
                    results.append({"source": "cognee", "content": str(item)})
            return results
        except Exception:
            logger.exception("Cognee recall failed; returning SQLite results only.")
            return []
