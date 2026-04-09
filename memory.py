"""
memory.py — Persistent SQLite storage for KOBRA.

Two tables:
  - conversations: every user/assistant turn
  - facts: key/value store for long-term facts the LLM explicitly saves
"""

import sqlite3
import logging
from datetime import datetime
from typing import Any

import config

logger = logging.getLogger(__name__)


class Memory:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_tables()
        logger.info("Memory initialised — DB: %s", config.DB_PATH)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _initialize_tables(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    key       TEXT    UNIQUE NOT NULL,
                    value     TEXT    NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    # ── Writes ─────────────────────────────────────────────────────────────────

    def save_conversation_turn(self, role: str, content: str) -> None:
        """Append one conversation turn (role: 'user' | 'assistant')."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversations (role, content) VALUES (?, ?)",
                (role, content),
            )

    def save_fact(self, key: str, value: str) -> None:
        """Upsert a named fact into long-term storage."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO facts (key, value, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value     = excluded.value,
                    timestamp = excluded.timestamp
                """,
                (key, value),
            )
        logger.info("Fact saved — %s: %s", key, value)

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = config.MEMORY_INJECT_LIMIT) -> list[dict[str, Any]]:
        """Return the last N conversation turns, oldest-first."""
        cur = self._conn.execute(
            """
            SELECT role, content, timestamp
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_all_facts(self) -> list[dict[str, Any]]:
        """Return all stored facts."""
        cur = self._conn.execute("SELECT key, value, timestamp FROM facts ORDER BY key")
        return [dict(r) for r in cur.fetchall()]

    def recall(self, query: str) -> list[dict[str, Any]]:
        """
        Search conversations and facts for the query string (case-insensitive LIKE).
        Returns a list of dicts with a 'source' field indicating which table matched.
        """
        like = f"%{query}%"
        results: list[dict[str, Any]] = []

        cur = self._conn.execute(
            "SELECT role, content, timestamp FROM conversations WHERE content LIKE ? ORDER BY id DESC LIMIT 10",
            (like,),
        )
        for row in cur.fetchall():
            results.append({**dict(row), "source": "conversation"})

        cur = self._conn.execute(
            "SELECT key, value, timestamp FROM facts WHERE key LIKE ? OR value LIKE ?",
            (like, like),
        )
        for row in cur.fetchall():
            results.append({**dict(row), "source": "fact"})

        return results

    # ── Maintenance ────────────────────────────────────────────────────────────

    def clear_conversations(self) -> None:
        """Delete all conversation history (facts are preserved)."""
        with self._conn:
            self._conn.execute("DELETE FROM conversations")
        logger.info("Conversation history cleared.")

    def format_for_injection(self) -> str:
        """
        Build the memory context block that gets inserted into the system prompt.
        """
        lines: list[str] = ["--- MEMORY CONTEXT ---"]

        recent = self.get_recent()
        if recent:
            lines.append(f"Recent conversation (last {len(recent)} turns):")
            for turn in recent:
                lines.append(f"[{turn['role']}]: {turn['content']}")
        else:
            lines.append("No recent conversation history.")

        facts = self.get_all_facts()
        if facts:
            lines.append("\nStored facts:")
            for f in facts:
                lines.append(f"- {f['key']}: {f['value']}")

        lines.append("--- END MEMORY ---")
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
