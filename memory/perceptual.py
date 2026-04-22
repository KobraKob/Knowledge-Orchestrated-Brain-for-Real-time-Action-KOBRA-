"""
memory/perceptual.py — PerceptualMemory for KOBRA v5.

Stores what exists now: RAG knowledge base + live API data.
All queries return fresh data. No caching (except RAG which has its own).

Live sources (optional — degrade gracefully if not configured):
  - RAG (ChromaDB) — indexed local files
  - Calendar (Google) — today's events
  - Email (Gmail) — unread messages
  - Filesystem — recently modified files
"""

import logging
import os
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

import config

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
            self.final_score = self.relevance_score * 0.5 + self.recency_score * 0.5


_RAG_TRIGGERS = frozenset({
    "my project", "my file", "my notes", "my document", "i wrote",
    "that code", "my script", "find my", "search my", "what did i",
    "recall my", "in my", "look up my", "the project", "my repo",
})


class PerceptualMemory:
    """
    Fresh-data memory layer. Queries RAG and optionally live APIs.
    All live API calls are gated — if client is None, silently skip.
    """

    def __init__(
        self,
        retriever=None,
        calendar_client=None,
        email_client=None,
    ) -> None:
        self._retriever = retriever
        self._calendar  = calendar_client
        self._email     = email_client

    def query(self, question: str, include_live: bool = False, limit: int = 5) -> list[MemoryResult]:
        """Fan out to all perceptual sources. Returns ranked MemoryResults."""
        results = []

        # RAG — only when triggered by keywords
        if self._retriever and self._should_query_rag(question):
            results.extend(self._query_rag(question, limit=3))

        if include_live:
            if self._calendar:
                results.extend(self._query_calendar())
            if self._email:
                results.extend(self._query_email())
            results.extend(self._query_recent_files())

        # Sort by final_score and cap
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:limit]

    def get_live_snapshot(self) -> dict:
        """
        Full current-state snapshot for morning briefing.
        Returns dict with calendar, unread_emails, recent_files, open_projects.
        """
        snapshot = {
            "calendar":      [],
            "unread_emails": [],
            "recent_files":  [],
            "open_projects": [],
            "rag_summary":   "",
        }
        try:
            if self._calendar:
                snapshot["calendar"] = self._get_calendar_today()
        except Exception as exc:
            logger.debug("[PERCEPTUAL] calendar snapshot failed: %s", exc)
        try:
            if self._email:
                snapshot["unread_emails"] = self._get_unread_emails()
        except Exception as exc:
            logger.debug("[PERCEPTUAL] email snapshot failed: %s", exc)
        try:
            snapshot["recent_files"]  = self._get_recent_files()
            snapshot["open_projects"] = self._get_open_projects()
        except Exception as exc:
            logger.debug("[PERCEPTUAL] filesystem snapshot failed: %s", exc)
        return snapshot

    # ── Private ───────────────────────────────────────────────────────────────

    def _should_query_rag(self, question: str) -> bool:
        q = question.lower()
        return any(t in q for t in _RAG_TRIGGERS)

    def _query_rag(self, question: str, limit: int = 3) -> list[MemoryResult]:
        results = []
        try:
            chunks = self._retriever.retrieve(question)
            for chunk in (chunks or [])[:limit]:
                text = chunk.get("text", str(chunk))[:400] if isinstance(chunk, dict) else str(chunk)[:400]
                results.append(MemoryResult(
                    content=f"[Knowledge base]: {text}",
                    source="perceptual",
                    subsource="rag",
                    relevance_score=0.85,
                    recency_score=0.6,
                ))
        except Exception as exc:
            logger.debug("[PERCEPTUAL] RAG query failed: %s", exc)
        return results

    def _query_calendar(self) -> list[MemoryResult]:
        results = []
        try:
            events = self._get_calendar_today()
            for e in events[:3]:
                results.append(MemoryResult(
                    content=f"[Calendar]: {e.get('title','Event')} at {e.get('time','')}",
                    source="perceptual",
                    subsource="calendar",
                    relevance_score=0.9,
                    recency_score=1.0,
                    timestamp=datetime.utcnow(),
                ))
        except Exception:
            pass
        return results

    def _query_email(self) -> list[MemoryResult]:
        results = []
        try:
            emails = self._get_unread_emails()
            for e in emails[:2]:
                results.append(MemoryResult(
                    content=f"[Email] from {e.get('sender','?')}: {e.get('subject','?')}",
                    source="perceptual",
                    subsource="email",
                    relevance_score=0.7,
                    recency_score=0.9,
                ))
        except Exception:
            pass
        return results

    def _query_recent_files(self) -> list[MemoryResult]:
        results = []
        try:
            files = self._get_recent_files(hours=24, limit=3)
            for f in files:
                results.append(MemoryResult(
                    content=f"[Recent file]: {f.get('path','?')} (modified {f.get('hours_ago',0):.1f}h ago)",
                    source="perceptual",
                    subsource="filesystem",
                    relevance_score=0.5,
                    recency_score=0.8,
                ))
        except Exception:
            pass
        return results

    def _get_calendar_today(self) -> list[dict]:
        """Fetch today's calendar events via integration client."""
        if not self._calendar:
            return []
        try:
            return self._calendar.get_events_today() or []
        except Exception:
            return []

    def _get_unread_emails(self) -> list[dict]:
        if not self._email:
            return []
        try:
            return self._email.get_unread(max_results=5) or []
        except Exception:
            return []

    def _get_recent_files(self, hours: int = 48, limit: int = 5) -> list[dict]:
        cutoff = datetime.now() - timedelta(hours=hours)
        results = []
        for folder in config.WATCHED_FOLDERS:
            if not os.path.exists(folder):
                continue
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__", "venv", ".venv")]
                for fname in files:
                    if fname.startswith("."):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                        if mtime > cutoff:
                            hours_ago = (datetime.now() - mtime).total_seconds() / 3600
                            results.append({"path": fpath, "hours_ago": round(hours_ago, 1), "mtime": mtime})
                    except Exception:
                        continue
                if len(results) >= limit * 3:
                    break
        results.sort(key=lambda x: x["hours_ago"])
        return results[:limit]

    def _get_open_projects(self) -> list[dict]:
        """Find project dirs with recent activity."""
        projects = []
        for folder in config.WATCHED_FOLDERS:
            if not os.path.exists(folder):
                continue
            try:
                for name in os.listdir(folder):
                    path = os.path.join(folder, name)
                    if not os.path.isdir(path):
                        continue
                    git_dir = os.path.join(path, ".git")
                    if not os.path.exists(git_dir):
                        continue
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(git_dir))
                        hours_ago = (datetime.now() - mtime).total_seconds() / 3600
                        if hours_ago < 72:
                            projects.append({
                                "name": name, "path": path,
                                "hours_ago": round(hours_ago, 1),
                            })
                    except Exception:
                        continue
            except Exception:
                continue
        return sorted(projects, key=lambda x: x["hours_ago"])[:5]
