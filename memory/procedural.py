"""
memory/procedural.py — ProceduralMemory for KOBRA v5.

Stores how to do things: routing strategies, tool success rates, corrections.
Wraps existing routing_memory.py and learning.py.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    content:        str
    source:         str
    subsource:      str
    relevance_score: float
    recency_score:  float
    final_score:    float = 0.0
    timestamp:      datetime | None = None
    metadata:       dict = field(default_factory=dict)

    def __post_init__(self):
        if self.final_score == 0.0:
            self.final_score = self.relevance_score * 0.6 + self.recency_score * 0.4


_STOPWORDS = frozenset({"what", "when", "where", "which", "that", "this", "have",
                         "does", "about", "from", "with", "into", "will", "your"})


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w{3,}\b", text.lower()) if w not in _STOPWORDS}


class ProceduralMemory:
    """
    Wraps RoutingMemory and LearningSystem with typed MemoryResult output.
    Falls back gracefully if underlying modules unavailable.
    """

    def __init__(self, routing_memory=None, learning_system=None) -> None:
        self._routing = routing_memory
        self._learning = learning_system

    def query(self, question: str, limit: int = 3) -> list[MemoryResult]:
        """Return past successful routing examples similar to this question."""
        results = []
        if not self._routing:
            return results
        try:
            few_shots_str = self._routing.get_few_shot_examples(question, limit=limit)
            if few_shots_str:
                results.append(MemoryResult(
                    content=few_shots_str,
                    source="procedural",
                    subsource="routing",
                    relevance_score=0.75,
                    recency_score=0.8,
                ))
        except Exception as exc:
            logger.debug("[PROCEDURAL] query failed: %s", exc)

        # Also check learning correction hints
        if self._learning:
            try:
                corrections = self._learning.get_routing_correction_context()
                if corrections:
                    results.append(MemoryResult(
                        content=corrections,
                        source="procedural",
                        subsource="corrections",
                        relevance_score=0.9,
                        recency_score=0.9,
                    ))
            except Exception:
                pass

        return results[:limit]

    def log_routing(self, question: str, route: str, outcome: str, duration: float = 0.0) -> None:
        if self._routing:
            try:
                self._routing.log_routing(question, route, "", outcome)
            except Exception:
                pass
        if self._learning:
            try:
                self._learning.log_tool_outcome(route, "routing", outcome == "success", duration)
            except Exception:
                pass

    def log_correction(self, original_route: str, corrected_route: str, transcript: str) -> None:
        if self._routing:
            try:
                self._routing.log_correction(transcript, original_route, corrected_route)
            except Exception:
                pass
        if self._learning:
            try:
                self._learning.log_routing_correction(transcript, original_route, corrected_route)
            except Exception:
                pass

    def get_tool_success_rates(self, agent: str) -> dict:
        if self._learning:
            try:
                return self._learning.get_agent_success_rates().get(agent, {})
            except Exception:
                pass
        return {}
