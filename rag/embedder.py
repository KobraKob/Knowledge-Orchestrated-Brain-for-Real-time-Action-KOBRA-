"""
rag/embedder.py — Text embedding via Ollama's nomic-embed-text model.

Runs locally on RTX 3050 — no API key, no internet, no cost.
Requires: ollama running + `ollama pull nomic-embed-text`
"""

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self._base_url = config.OLLAMA_BASE_URL
        self._model = config.RAG_EMBED_MODEL
        self._verified = False

    def _verify_model_available(self) -> None:
        if self._verified:
            return
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self._model in m for m in models):
                logger.info("[RAG] Pulling %s …", self._model)
                subprocess.run(["ollama", "pull", self._model], check=True, timeout=300)
            self._verified = True
        except requests.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running. Start it with: ollama serve\n"
                f"Then run: ollama pull {self._model}"
            )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        self._verify_model_available()
        response = requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in parallel.
        Returns embeddings in the same order as the input list.
        """
        self._verify_model_available()
        results: dict[int, list[float]] = {}

        def embed_one(idx_text):
            idx, text = idx_text
            return idx, self.embed(text)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(embed_one, (i, t)): i
                       for i, t in enumerate(texts)}
            for future in as_completed(futures):
                try:
                    idx, embedding = future.result()
                    results[idx] = embedding
                except Exception as exc:
                    idx = futures[future]
                    logger.warning("Embedding failed for index %d: %s", idx, exc)
                    results[idx] = []  # empty vector as fallback

        return [results[i] for i in range(len(texts))]
