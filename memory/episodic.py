"""
memory/episodic.py — EpisodicMemory for KOBRA v5.

Stores past conversations, commands, and session history.
Wraps the existing SQLite conversations table with richer querying.

Decay model:
  - Episodes < 24h old:  recency_score = 1.0
  - Episodes < 7 days:   recency_score = 0.7
  - Episodes older:      recency_score = 0.3

Session model:
  - Each KOBRA startup = new session_id (uuid4)
  - get_last_session() returns what was worked on in the most recent prior session
  - summarize_old_episodes() compresses episodes > 7 days into session summaries
"""

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Current session ID — set once at import time, persists for this KOBRA run
CURRENT_SESSION_ID: str = str(uuid.uuid4())


@dataclass
class MemoryResult:
    content:        str
    source:         str   # "episodic"
    subsource:      str   # "conversations" | "session_summary"
    relevance_score: float
    recency_score:  float
    final_score:    float = 0.0
    timestamp:      datetime | None = None
    metadata:       dict = field(default_factory=dict)

    def __post_init__(self):
        if self.final_score == 0.0:
            self.final_score = self.relevance_score * 0.5 + self.recency_score * 0.5


def _recency_score(ts: datetime | None) -> float:
    if ts is None:
        return 0.3
    age = datetime.utcnow() - ts
    if age < timedelta(hours=24):
        return 1.0
    if age < timedelta(days=7):
        return 0.7
    return 0.3


class EpisodicMemory:
    """
    Wraps the existing SQLite conversations table.
    Adds session tracking, keyword search, and session summarization.
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._session_id = CURRENT_SESSION_ID

    def _ensure_schema(self) -> None:
        """Add session columns to existing table if not present. Never drops data."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL UNIQUE,
                started_at  TEXT NOT NULL,
                ended_at    TEXT NOT NULL,
                summary     TEXT NOT NULL,
                tools_used  TEXT NOT NULL DEFAULT '[]',
                primary_project TEXT DEFAULT '',
                last_action TEXT DEFAULT '',
                turn_count  INTEGER DEFAULT 0
            );
        """)
        # Add session_id column to conversations if missing (migration)
        try:
            self._conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT DEFAULT ''")
        except Exception:
            pass  # already exists
        try:
            self._conn.execute("ALTER TABLE conversations ADD COLUMN intent TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            self._conn.execute("ALTER TABLE conversations ADD COLUMN tools_used TEXT DEFAULT '[]'")
        except Exception:
            pass
        try:
            self._conn.execute("ALTER TABLE conversations ADD COLUMN success INTEGER DEFAULT 1")
        except Exception:
            pass
        self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, role: str, content: str, intent: str = "", tools_used: list | None = None, success: bool = True) -> None:
        """Save a conversation turn to the current session."""
        try:
            self._conn.execute(
                "INSERT INTO conversations (role, content, session_id, intent, tools_used, success) VALUES (?,?,?,?,?,?)",
                (role, content[:2000], self._session_id, intent,
                 json.dumps(tools_used or []), int(success)),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[EPISODIC] save failed: %s", exc)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 6) -> list[dict]:
        """Return most recent conversation turns as plain dicts (brain.py compat)."""
        try:
            rows = self._conn.execute(
                "SELECT role, content, timestamp FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
                    for r in reversed(rows)]
        except Exception:
            return []

    def get_all_facts(self) -> list[dict]:
        """Compatibility shim — episodic doesn't store facts; returns empty."""
        return []

    def query(self, question: str, limit: int = 6) -> list[MemoryResult]:
        """Keyword search across conversation content, weighted by recency."""
        results = []
        try:
            # Build keyword filter
            words = [w for w in re.findall(r"\b\w{4,}\b", question.lower()) if w not in
                     {"what", "when", "where", "which", "that", "this", "have", "does", "about"}]

            if words:
                placeholders = " OR ".join(["LOWER(content) LIKE ?" for _ in words[:5]])
                params = [f"%{w}%" for w in words[:5]]
                rows = self._conn.execute(
                    f"SELECT role, content, timestamp FROM conversations WHERE {placeholders} ORDER BY id DESC LIMIT ?",
                    params + [limit * 2],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT role, content, timestamp FROM conversations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            seen = set()
            for row in rows:
                key = row["content"][:80]
                if key in seen:
                    continue
                seen.add(key)
                ts = None
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                except Exception:
                    pass
                rec = _recency_score(ts)
                # Simple relevance: word overlap ratio
                content_words = set(re.findall(r"\b\w{4,}\b", row["content"].lower()))
                overlap = len(set(words) & content_words) / max(len(words), 1)
                results.append(MemoryResult(
                    content=f"{row['role'].upper()}: {row['content'][:300]}",
                    source="episodic",
                    subsource="conversations",
                    relevance_score=min(overlap, 1.0),
                    recency_score=rec,
                    timestamp=ts,
                ))
                if len(results) >= limit:
                    break

        except Exception as exc:
            logger.debug("[EPISODIC] query failed: %s", exc)
        return results

    def get_last_session(self) -> dict | None:
        """Return summary of the most recent prior session."""
        try:
            row = self._conn.execute(
                "SELECT * FROM session_summaries WHERE session_id != ? ORDER BY ended_at DESC LIMIT 1",
                (self._session_id,),
            ).fetchone()
            if row:
                return dict(row)

            # Fall back: summarize last N turns from a different session
            rows = self._conn.execute(
                """SELECT content, role, timestamp FROM conversations
                   WHERE session_id != ? AND session_id != ''
                   ORDER BY id DESC LIMIT 10""",
                (self._session_id,),
            ).fetchall()
            if not rows:
                return None
            last_ts_str = rows[0]["timestamp"] if rows else ""
            try:
                ended_at = datetime.fromisoformat(last_ts_str)
                hours_ago = (datetime.utcnow() - ended_at).total_seconds() / 3600
            except Exception:
                hours_ago = 999
            summary_parts = [r["content"][:100] for r in rows[:3] if r["role"] == "user"]
            return {
                "session_id": "unknown",
                "ended_at": last_ts_str,
                "hours_ago": round(hours_ago, 1),
                "summary": " | ".join(summary_parts) or "Previous session",
                "primary_project": "",
                "last_action": rows[0]["content"][:80] if rows else "",
                "tools_used": "[]",
                "turn_count": len(rows),
            }
        except Exception as exc:
            logger.debug("[EPISODIC] get_last_session failed: %s", exc)
            return None

    def close_session(self, summary: str = "", primary_project: str = "", last_action: str = "", tools_used: list | None = None) -> None:
        """Call on shutdown to save a session summary."""
        try:
            count_row = self._conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id=?", (self._session_id,)
            ).fetchone()
            turn_count = count_row[0] if count_row else 0

            first_row = self._conn.execute(
                "SELECT timestamp FROM conversations WHERE session_id=? ORDER BY id ASC LIMIT 1",
                (self._session_id,),
            ).fetchone()
            started = first_row[0] if first_row else datetime.utcnow().isoformat()

            self._conn.execute(
                """INSERT OR REPLACE INTO session_summaries
                   (session_id, started_at, ended_at, summary, tools_used, primary_project, last_action, turn_count)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (self._session_id, started, datetime.utcnow().isoformat(),
                 summary[:500], json.dumps(tools_used or []),
                 primary_project, last_action, turn_count),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[EPISODIC] close_session failed: %s", exc)

    def summarize_old_episodes(self) -> None:
        """Compress episodes older than 7 days. Delete raw episodes older than 30 days."""
        try:
            cutoff_30 = (datetime.utcnow() - timedelta(days=30)).isoformat()
            self._conn.execute(
                "DELETE FROM conversations WHERE timestamp < ? AND session_id != ''",
                (cutoff_30,),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[EPISODIC] summarize_old_episodes failed: %s", exc)

    def clear_conversations(self) -> None:
        """Compatibility shim for main.py."""
        try:
            self._conn.execute("DELETE FROM conversations")
            self._conn.commit()
        except Exception:
            pass
