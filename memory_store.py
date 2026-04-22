"""
memory.py — Persistent SQLite storage for KOBRA.

Two concerns:
  - Conversation history  → managed by ConversationSummaryBuffer
                            (rolling window + Groq-powered summarisation)
  - Long-term facts       → plain key/value SQLite table (never summarised)

ConversationSummaryBuffer keeps the last 8 turns verbatim.
When total turns exceed 20, older turns are compressed into a rolling
summary stored alongside the raw turns. Long sessions never blow the
context window — recent context stays sharp, old context is summarised.
"""

import sqlite3
import logging
from typing import Any

from groq import Groq

import config
from conversation_memory import ConversationSummaryBuffer

logger = logging.getLogger(__name__)


class Memory:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_tables()

        # ConversationSummaryBuffer — wraps the conversations table
        _groq = Groq(api_key=config.GROQ_API_KEY)
        self._csb = ConversationSummaryBuffer(self._conn, _groq)

        logger.info("Memory initialised — DB: %s", config.DB_PATH)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _initialize_tables(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    is_summary INTEGER DEFAULT 0,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    key       TEXT    UNIQUE NOT NULL,
                    value     TEXT    NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    # ── Conversation writes ────────────────────────────────────────────────────

    def save_conversation_turn(self, role: str, content: str) -> None:
        """
        Append one conversation turn and trigger summarisation if needed.
        After every save, ConversationSummaryBuffer checks whether old turns
        need to be compacted — completely transparent to the caller.
        """
        self._csb.add_turn(role, content)
        # Summarise asynchronously — fire and forget within the same call.
        # In practice this is fast (<1s) and only triggers every ~12 turns.
        self._csb.maybe_summarise()

    # ── Fact writes ────────────────────────────────────────────────────────────

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

    # ── Conversation reads ─────────────────────────────────────────────────────

    def get_recent(self, limit: int = config.MEMORY_INJECT_LIMIT) -> list[dict[str, Any]]:
        """
        Return recent raw conversation turns, oldest-first.
        Used by brain.py to inject history into Groq messages.
        Does NOT include the summary row — use get_context() for that.
        """
        return self._csb.get_recent_raw(limit=limit)

    def get_context(self) -> list[dict[str, Any]]:
        """
        Return full context for LLM injection:
          [summary system message (if any)] + [last 8 raw turns]
        This is the ConversationSummaryBuffer's primary output.
        """
        return self._csb.get_context()

    # ── Fact reads ─────────────────────────────────────────────────────────────

    def get_all_facts(self) -> list[dict[str, Any]]:
        """Return all stored long-term facts."""
        cur = self._conn.execute("SELECT key, value, timestamp FROM facts ORDER BY key")
        return [dict(r) for r in cur.fetchall()]

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Search conversations and facts for the query string."""
        like = f"%{query}%"
        results: list[dict[str, Any]] = []

        cur = self._conn.execute(
            """SELECT role, content, timestamp FROM conversations
               WHERE content LIKE ? AND is_summary = 0
               ORDER BY id DESC LIMIT 10""",
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
        """Delete all conversation history and summaries (facts preserved)."""
        self._csb.clear()
        logger.info("Conversation history cleared.")

    def format_for_injection(self) -> str:
        """Build a memory context string for legacy callers."""
        lines: list[str] = ["--- MEMORY CONTEXT ---"]
        recent = self.get_recent()
        if recent:
            lines.append(f"Recent conversation ({len(recent)} turns):")
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
