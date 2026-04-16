"""
memory_router.py — Unified memory interface for KOBRA v4.

Single query interface over all 5 memory systems.
The Planner, Context Builder, and Agents all talk to this instead of
querying each DB separately.

Query strategy:
  - Conversation memory: always injected (most recent turns)
  - Semantic memory: always injected (user profile facts)
  - Episodic memory: injected only when relevant (keyword match)
  - RAG knowledge: injected only when query contains file/project/doc references
  - Routing memory: injected only into decomposition calls

Each system has a token budget. Total context is capped to avoid bloat.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Token budget per memory system (rough char estimate: 1 token ≈ 4 chars)
_BUDGET = {
    "conversation": 800,    # ~200 tokens — recent turns
    "semantic":     600,    # ~150 tokens — user profile
    "episodic":     400,    # ~100 tokens — relevant past events
    "rag":          1200,   # ~300 tokens — file knowledge
    "routing":      300,    # ~75 tokens  — routing hints
}

# RAG trigger keywords — only query ChromaDB when these appear
_RAG_TRIGGERS = (
    "my project", "my file", "my notes", "my document", "i wrote",
    "that code", "my script", "find my", "search my", "what did i",
    "recall my", "in my", "look up my", "the project", "my repo",
)


class MemoryRouter:
    """
    Unified memory query interface.

    Usage:
        router = MemoryRouter(memory, learning, routing_memory, retriever)
        context = router.build_context(query, include_rag=True)
    """

    def __init__(
        self,
        conversation_memory,   # memory.Memory instance
        learning_system,       # learning.LearningSystem instance
        routing_memory=None,   # routing_memory.RoutingMemory instance (optional)
        retriever=None,        # RAG retriever (optional)
    ) -> None:
        self._conv = conversation_memory
        self._learning = learning_system
        self._routing = routing_memory
        self._retriever = retriever

    def build_context(
        self,
        query: str,
        include_routing: bool = False,
        max_conv_turns: int = 6,
    ) -> str:
        """
        Build a complete context string for a query.
        Each memory system contributes up to its token budget.

        Returns a single string ready to prepend to a prompt.
        """
        parts = []

        # 1. Semantic memory (always — user profile)
        semantic = self._get_semantic(query)
        if semantic:
            parts.append(self._truncate(semantic, _BUDGET["semantic"]))

        # 2. Episodic memory (relevant only)
        episodic = self._get_episodic(query)
        if episodic:
            parts.append(self._truncate(episodic, _BUDGET["episodic"]))

        # 3. Conversation memory (always — recent turns)
        conv = self._get_conversation(max_conv_turns)
        if conv:
            parts.append(self._truncate(conv, _BUDGET["conversation"]))

        # 4. RAG knowledge (only when triggered)
        if self._should_query_rag(query):
            rag = self._get_rag(query)
            if rag:
                parts.append(self._truncate(rag, _BUDGET["rag"]))

        # 5. Routing memory (only for decomposition)
        if include_routing and self._routing:
            routing = self._routing.get_few_shot_examples(query, limit=3)
            if routing:
                parts.append(self._truncate(routing, _BUDGET["routing"]))

        return "\n".join(parts)

    def build_decomposition_context(self, query: str, max_conv_turns: int = 4) -> str:
        """Context specifically for the orchestrator decomposition call."""
        return self.build_context(query, include_routing=True, max_conv_turns=max_conv_turns)

    def build_agent_context(self, query: str, agent_name: str) -> str:
        """
        Context for a specific agent — more targeted than full context.
        Skips routing memory, uses smaller conversation window.
        """
        parts = []

        semantic = self._get_semantic(query)
        if semantic:
            parts.append(self._truncate(semantic, _BUDGET["semantic"]))

        # Only inject episodic if relevant
        episodic = self._get_episodic(query)
        if episodic:
            parts.append(self._truncate(episodic, _BUDGET["episodic"]))

        # RAG only for knowledge/dev/research agents and when triggered
        if agent_name in ("knowledge", "dev", "research", "interpreter") and self._should_query_rag(query):
            rag = self._get_rag(query)
            if rag:
                parts.append(self._truncate(rag, _BUDGET["rag"]))

        # Short conversation window for agents
        conv = self._get_conversation(3)
        if conv:
            parts.append(self._truncate(conv, _BUDGET["conversation"] // 2))

        return "\n".join(parts)

    def store_episode_from_result(self, transcript: str, agent: str, output: str, success: bool) -> None:
        """
        After a task completes, store an episodic summary.
        Also runs semantic inference.
        """
        outcome = "success" if success else "failure"
        # Build a human-readable episode summary
        summary = f"You asked KOBRA to: {transcript[:100]}. Agent {agent} {'succeeded' if success else 'failed'}."
        if output and len(output) > 10:
            summary += f" Result: {output[:100]}"

        # Extract keywords from transcript
        keywords = list(set(re.findall(r"\b[a-zA-Z]{4,}\b", transcript.lower())))[:8]

        self._learning.store_episode(summary, agent, keywords, outcome)
        self._learning.infer_semantic_from_episode(transcript, agent, outcome)

    def store_semantic_fact(self, key: str, value: str, category: str) -> None:
        """Explicitly store a semantic fact."""
        self._learning.store_semantic(key, value, category)

    def cleanup(self) -> None:
        """Periodic cleanup — remove expired episodes."""
        self._learning.cleanup_expired_episodes()

    # ── Private retrieval methods ─────────────────────────────────────────────

    def _get_semantic(self, query: str) -> str:
        try:
            return self._learning.get_semantic_context()
        except Exception:
            return ""

    def _get_episodic(self, query: str) -> str:
        try:
            return self._learning.get_relevant_episodes(query)
        except Exception:
            return ""

    def _get_conversation(self, limit: int) -> str:
        try:
            turns = self._conv.get_recent(limit=limit)
            if not turns:
                return ""
            lines = ["Recent conversation:"]
            for t in turns:
                role = "User" if t["role"] == "user" else "KOBRA"
                lines.append(f"  {role}: {t['content'][:200]}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_rag(self, query: str) -> str:
        try:
            if not self._retriever:
                return ""
            results = self._retriever.retrieve(query)
            if not results:
                return ""
            lines = ["From your knowledge base:"]
            for chunk in results[:3]:
                if isinstance(chunk, dict):
                    lines.append(f"  {chunk.get('text', str(chunk))[:200]}")
                else:
                    lines.append(f"  {str(chunk)[:200]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("[MEMORY_ROUTER] RAG retrieval failed: %s", exc)
            return ""

    def _should_query_rag(self, query: str) -> bool:
        """Fast heuristic — only query RAG when likely relevant."""
        q = query.lower()
        return any(trigger in q for trigger in _RAG_TRIGGERS)

    @staticmethod
    def _truncate(text: str, char_limit: int) -> str:
        """Truncate text to char_limit, preserving whole lines."""
        if len(text) <= char_limit:
            return text
        truncated = text[:char_limit]
        last_newline = truncated.rfind("\n")
        if last_newline > char_limit // 2:
            return truncated[:last_newline] + "\n[...truncated]"
        return truncated + "...[truncated]"
