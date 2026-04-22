"""
memory/router.py — Unified Memory Router for KOBRA v5.

Single query interface over all 4 typed memory layers.
Every module that needs context calls MemoryRouter.query() — nothing else.

Algorithm:
  1. Check LRU cache (60s TTL, 50 entries max)
  2. Fan out to all 4 layers in parallel (ThreadPoolExecutor)
  3. Score each result: final_score = relevance * RELEVANCE_WEIGHT[intent] + recency * RECENCY_WEIGHT[intent]
  4. Deduplicate (content similarity > 0.85 → drop)
  5. Trim to token budget (config.MEMORY_TOKEN_BUDGET, default 800 tokens)
  6. Format into prompt block
  7. Cache and return MemoryBundle
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── Intent-based scoring weights ──────────────────────────────────────────────

_RELEVANCE_WEIGHT = {
    "direct":      0.3,   # facts/semantics matter more than recent history
    "action":      0.5,   # balanced
    "planning":    0.4,
    "search":      0.4,
    "real_time":   0.2,   # recency dominates for real-time queries
    "morning":     0.45,  # morning briefing: balanced
}

_RECENCY_WEIGHT = {
    "direct":      0.7,
    "action":      0.5,
    "planning":    0.6,
    "search":      0.6,
    "real_time":   0.8,
    "morning":     0.55,
}

# ── Token budget (chars ÷ 4 ≈ tokens) ────────────────────────────────────────
_DEFAULT_TOKEN_BUDGET = getattr(config, "MEMORY_TOKEN_BUDGET", 800)
_CHARS_PER_TOKEN = 4
_CACHE_TTL = getattr(config, "MEMORY_CACHE_TTL", 60)
_CACHE_MAX  = 50


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class QueryContext:
    intent:       str = "action"       # "direct" | "action" | "search" | "planning" | "real_time"
    task_type:    str = "general"      # "dev" | "media" | "web" | "research" | "conversation"
    time_of_day:  str = "day"          # "morning" | "afternoon" | "evening" | "night"
    urgency:      str = "normal"       # "real_time" | "normal" | "background"
    include_live: bool = False         # whether to hit live APIs (calendar, email)


@dataclass
class MemoryBundle:
    episodic:    list   # list[MemoryResult] — past conversations
    semantic:    list   # list[MemoryResult] — stored facts
    procedural:  list   # list[MemoryResult] — routing hints
    perceptual:  list   # list[MemoryResult] — RAG + live data
    formatted:   str    # pre-formatted string for prompt injection
    total_tokens: int   # estimated token count


# ── LRU Cache ────────────────────────────────────────────────────────────────

class _LRUCache:
    def __init__(self, maxsize: int = _CACHE_MAX, ttl: int = _CACHE_TTL) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, val = entry
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            return val

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            if len(self._cache) >= self._maxsize:
                # Evict oldest
                oldest = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest]
            self._cache[key] = (time.time(), val)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ── Similarity (fast, no embeddings needed for dedup) ─────────────────────────

def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w{4,}\b", text.lower()))

def _similarity(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)


# ── Main class ────────────────────────────────────────────────────────────────

class MemoryRouter:
    """
    Unified interface over all 4 memory layers.
    Drop-in replacement for the flat memory_router.py used by brain.py and orchestrator.py.
    """

    def __init__(
        self,
        episodic=None,
        semantic=None,
        procedural=None,
        perceptual=None,
        # legacy flat-router compat params (ignored, kept for signature compat)
        conversation_memory=None,
        learning_system=None,
        routing_memory=None,
        retriever=None,
    ) -> None:
        # Accept typed layers directly OR build from legacy flat objects
        self._episodic   = episodic
        self._semantic   = semantic
        self._procedural = procedural
        self._perceptual = perceptual

        # Legacy compat: if typed layers not provided, wrap flat objects
        if self._episodic is None and conversation_memory is not None:
            self._episodic = _LegacyEpisodicWrapper(conversation_memory)
        if self._semantic is None and learning_system is not None:
            self._semantic = _LegacySemanticWrapper(learning_system)
        if self._procedural is None and routing_memory is not None:
            self._procedural = _LegacyProceduralWrapper(routing_memory)
        if self._perceptual is None and retriever is not None:
            self._perceptual = _LegacyPerceptualWrapper(retriever)

        self._cache = _LRUCache()
        self._routing = routing_memory  # kept for get_few_shot_examples compat

    # ── Public query interface ────────────────────────────────────────────────

    def query(
        self,
        question: str,
        context: QueryContext | None = None,
        include_routing: bool = False,
        max_conv_turns: int = 6,
    ) -> MemoryBundle:
        """
        Main query method. Returns a MemoryBundle with formatted context string.
        Caches results for 60 seconds.
        """
        if context is None:
            context = QueryContext()

        cache_key = f"{question[:80]}|{context.intent}|{context.include_live}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Fan out to all layers in parallel (5s timeout per layer)
        episodic_results   = []
        semantic_results   = []
        procedural_results = []
        perceptual_results = []

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {}
            if self._episodic:
                futures["episodic"]   = ex.submit(self._episodic.query, question, max_conv_turns)
            if self._semantic:
                futures["semantic"]   = ex.submit(self._semantic.query, question, 10)
            if self._procedural and include_routing:
                futures["procedural"] = ex.submit(self._procedural.query, question, 3)
            if self._perceptual:
                futures["perceptual"] = ex.submit(
                    self._perceptual.query, question, context.include_live, 5
                )

            for layer, fut in futures.items():
                try:
                    res = fut.result(timeout=5)
                    if layer == "episodic":
                        episodic_results = res or []
                    elif layer == "semantic":
                        semantic_results = res or []
                    elif layer == "procedural":
                        procedural_results = res or []
                    elif layer == "perceptual":
                        perceptual_results = res or []
                except Exception as exc:
                    logger.debug("[ROUTER] Layer %s failed: %s", layer, exc)

        # Score all results using intent weights
        rel_w = _RELEVANCE_WEIGHT.get(context.intent, 0.5)
        rec_w = _RECENCY_WEIGHT.get(context.intent, 0.5)
        all_results = episodic_results + semantic_results + procedural_results + perceptual_results
        for r in all_results:
            r.final_score = r.relevance_score * rel_w + r.recency_score * rec_w

        # Deduplicate by content similarity
        deduped = self._deduplicate(all_results)

        # Semantic always comes first (highest priority)
        semantic_deduped   = [r for r in deduped if r.source == "semantic"]
        remaining          = [r for r in deduped if r.source != "semantic"]
        remaining.sort(key=lambda r: r.final_score, reverse=True)

        # Trim to token budget
        budget_chars = _DEFAULT_TOKEN_BUDGET * _CHARS_PER_TOKEN
        selected = []
        used_chars = 0
        for r in semantic_deduped + remaining:
            if used_chars + len(r.content) > budget_chars:
                break
            selected.append(r)
            used_chars += len(r.content)

        # Re-split by layer
        ep  = [r for r in selected if r.source == "episodic"]
        sem = [r for r in selected if r.source == "semantic"]
        pro = [r for r in selected if r.source == "procedural"]
        per = [r for r in selected if r.source == "perceptual"]

        formatted = self._format(sem, ep, pro, per)
        bundle = MemoryBundle(
            episodic=ep, semantic=sem, procedural=pro, perceptual=per,
            formatted=formatted, total_tokens=used_chars // _CHARS_PER_TOKEN,
        )
        self._cache.set(cache_key, bundle)
        return bundle

    def build_context(
        self,
        query: str,
        include_routing: bool = False,
        max_conv_turns: int = 6,
    ) -> str:
        """Legacy flat-router compat. Returns formatted string."""
        bundle = self.query(
            query,
            context=QueryContext(intent="action", include_live=False),
            include_routing=include_routing,
            max_conv_turns=max_conv_turns,
        )
        return bundle.formatted

    def build_decomposition_context(self, query: str, max_conv_turns: int = 4) -> str:
        """Context for orchestrator decomposition — includes routing hints."""
        bundle = self.query(
            query,
            context=QueryContext(intent="planning"),
            include_routing=True,
            max_conv_turns=max_conv_turns,
        )
        return bundle.formatted

    def build_agent_context(self, query: str, agent_name: str) -> str:
        """Targeted context for a specific agent."""
        include_rag = agent_name in ("knowledge", "dev", "research", "interpreter")
        bundle = self.query(
            query,
            context=QueryContext(intent="action", include_live=False),
            include_routing=False,
            max_conv_turns=3,
        )
        return bundle.formatted

    def query_proactive(self, scanner_name: str) -> list:
        """For proactive/morning briefing — bypasses cache, includes live data."""
        self._cache.clear()
        bundle = self.query(
            "morning briefing current state",
            context=QueryContext(intent="morning", include_live=True),
            include_routing=False,
            max_conv_turns=4,
        )
        all_results = bundle.episodic + bundle.semantic + bundle.procedural + bundle.perceptual
        return all_results

    def store_episode_from_result(
        self, transcript: str, agent: str, output: str, success: bool
    ) -> None:
        """Store an episodic memory after task completion."""
        if self._episodic and hasattr(self._episodic, "save"):
            try:
                role = "assistant"
                summary = f"[{agent}] {output[:200]}" if output else f"[{agent}] completed"
                self._episodic.save(role, summary, intent=agent, success=success)
            except Exception:
                pass
        if self._semantic and hasattr(self._semantic, "infer_and_update"):
            try:
                self._semantic.infer_and_update({"transcript": transcript, "agent": agent})
            except Exception:
                pass
        self._cache.clear()

    def store_semantic_fact(self, key: str, value: str, category: str) -> None:
        if self._semantic and hasattr(self._semantic, "save_fact"):
            self._semantic.save_fact(key, value, category)

    def get_few_shot_examples(self, query: str, limit: int = 3) -> str:
        """Legacy compat — delegates to routing memory."""
        if self._routing and hasattr(self._routing, "get_few_shot_examples"):
            try:
                return self._routing.get_few_shot_examples(query, limit=limit)
            except Exception:
                pass
        if self._procedural:
            results = self._procedural.query(query, limit=limit)
            return results[0].content if results else ""
        return ""

    def forget(self, query: str, memory_type: str = "all") -> int:
        """Delete matching entries from specified layers."""
        count = 0
        # Currently: clear episodic conversations matching query
        if memory_type in ("all", "episodic") and self._episodic:
            try:
                if hasattr(self._episodic, "_conn"):
                    words = re.findall(r"\b\w{4,}\b", query.lower())
                    if words:
                        ph = " OR ".join(["LOWER(content) LIKE ?" for _ in words[:3]])
                        params = [f"%{w}%" for w in words[:3]]
                        cur = self._episodic._conn.execute(
                            f"DELETE FROM conversations WHERE {ph}", params
                        )
                        count += cur.rowcount
                        self._episodic._conn.commit()
            except Exception:
                pass
        self._cache.clear()
        return count

    def cleanup(self) -> None:
        """Periodic maintenance."""
        if self._episodic and hasattr(self._episodic, "summarize_old_episodes"):
            self._episodic.summarize_old_episodes()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _deduplicate(self, results: list) -> list:
        deduped = []
        for candidate in results:
            is_dup = any(
                _similarity(candidate.content, existing.content) > 0.85
                for existing in deduped
            )
            if not is_dup:
                deduped.append(candidate)
        return deduped

    def _format(self, semantic, episodic, procedural, perceptual) -> str:
        parts = []
        if semantic:
            lines = ["[User profile]"]
            for r in semantic[:8]:
                lines.append(f"  {r.content}")
            parts.append("\n".join(lines))
        if episodic:
            lines = ["[Recent context]"]
            for r in episodic[:4]:
                lines.append(f"  {r.content[:200]}")
            parts.append("\n".join(lines))
        if procedural:
            lines = ["[Routing hints]"]
            for r in procedural[:2]:
                lines.append(f"  {r.content[:200]}")
            parts.append("\n".join(lines))
        if perceptual:
            lines = ["[Current knowledge]"]
            for r in perceptual[:3]:
                lines.append(f"  {r.content[:200]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ── Legacy wrappers (used when typed layers not available) ─────────────────

    # See _Legacy*Wrapper classes below


# ── Legacy compatibility wrappers ─────────────────────────────────────────────
# These allow MemoryRouter to work with the existing flat objects from brain.py
# while the typed memory layers are gradually wired in.

class _LegacyEpisodicWrapper:
    def __init__(self, memory): self._m = memory
    def query(self, q, limit=6):
        from memory.episodic import MemoryResult
        try:
            turns = self._m.get_recent(limit=limit)
            results = []
            for t in turns:
                results.append(MemoryResult(
                    content=f"{t['role'].upper()}: {t['content'][:200]}",
                    source="episodic", subsource="conversations",
                    relevance_score=0.5, recency_score=0.8,
                ))
            return results
        except Exception:
            return []
    def save(self, role, content, **kw):
        try: self._m.save(role, content)
        except Exception: pass


class _LegacySemanticWrapper:
    def __init__(self, learning): self._l = learning
    def query(self, q, limit=10):
        from memory.semantic import MemoryResult
        results = []
        try:
            ctx = self._l.get_personalization_context()
            if ctx:
                results.append(MemoryResult(
                    content=ctx, source="semantic", subsource="facts",
                    relevance_score=0.8, recency_score=0.9,
                ))
        except Exception: pass
        return results
    def infer_and_update(self, ep):
        try: self._l.extract_vocabulary(ep.get("transcript",""))
        except Exception: pass
    def save_fact(self, key, value, category="general", **kw):
        try: self._l.store_semantic(key, value, category)
        except Exception: pass
    def get_all_preferences(self):
        return {}


class _LegacyProceduralWrapper:
    def __init__(self, routing): self._r = routing
    def query(self, q, limit=3):
        from memory.procedural import MemoryResult
        try:
            s = self._r.get_few_shot_examples(q, limit=limit)
            if s:
                return [MemoryResult(content=s, source="procedural", subsource="routing",
                                     relevance_score=0.75, recency_score=0.8)]
        except Exception: pass
        return []


class _LegacyPerceptualWrapper:
    def __init__(self, retriever): self._r = retriever
    def query(self, q, include_live=False, limit=5):
        from memory.perceptual import MemoryResult
        results = []
        try:
            chunks = self._r.retrieve(q)
            for chunk in (chunks or [])[:limit]:
                text = chunk.get("text", str(chunk))[:300] if isinstance(chunk, dict) else str(chunk)[:300]
                results.append(MemoryResult(
                    content=f"[Knowledge base]: {text}",
                    source="perceptual", subsource="rag",
                    relevance_score=0.85, recency_score=0.6,
                ))
        except Exception: pass
        return results
    def get_live_snapshot(self): return {}
