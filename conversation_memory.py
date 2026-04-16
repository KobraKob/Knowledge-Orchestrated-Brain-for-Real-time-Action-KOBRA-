"""
conversation_memory.py — ConversationSummaryBuffer for KOBRA.

Implements the same pattern as LangChain's ConversationSummaryBufferMemory
but natively on our existing SQLite schema — no Pydantic v1 / Python 3.14 issues.

How it works:
  - Keeps the last RECENT_TURNS turns verbatim in the conversations table.
  - When the raw turn count exceeds SUMMARY_THRESHOLD, older turns are
    collapsed into a rolling summary stored as a special 'summary' role row.
  - get_context() returns: [summary block] + [last RECENT_TURNS turns]
    — giving the LLM recent verbatim context AND long-term awareness
    without ever blowing the context window.

The facts table is completely unchanged — long-term facts are separate.
"""

import logging
import sqlite3
from typing import Any

from groq import Groq

import config

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# How many recent turns to keep verbatim (never summarised)
RECENT_TURNS: int = 8

# When total raw turns exceed this, trigger a summarisation pass
SUMMARY_THRESHOLD: int = 20

# Groq prompt for generating the rolling summary
_SUMMARY_SYSTEM = (
    "You are a memory summarizer for a voice assistant called KOBRA. "
    "Given a list of conversation turns, write a compact third-person summary "
    "that preserves: key facts mentioned, decisions made, tasks completed, "
    "and any user preferences expressed. "
    "Write 3-5 sentences max. Plain prose, no bullets, no headers."
)


class ConversationSummaryBuffer:
    """
    Drop-in upgrade for Memory.conversations.
    Wraps the same SQLite DB with a summarisation layer.

    Usage:
        csb = ConversationSummaryBuffer(conn, groq_client)
        csb.add_turn("user", "What's the weather like?")
        csb.add_turn("assistant", "It's 28°C and sunny in Mumbai.")
        messages = csb.get_context()   # → list[dict] for Groq messages API
        csb.maybe_summarise()           # call after each turn to compact if needed
    """

    def __init__(self, conn: sqlite3.Connection, groq_client: Groq) -> None:
        self._conn = conn
        self._groq = groq_client
        self._ensure_schema()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Add summary_text column to conversations if not present."""
        with self._conn:
            # conversations table already exists — just add the is_summary flag
            try:
                self._conn.execute(
                    "ALTER TABLE conversations ADD COLUMN is_summary INTEGER DEFAULT 0"
                )
                logger.info("[CSB] Added is_summary column to conversations table.")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ── Writes ─────────────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Append a raw conversation turn."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversations (role, content, is_summary) VALUES (?, ?, 0)",
                (role, content),
            )

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_context(self, recent_limit: int = RECENT_TURNS) -> list[dict[str, Any]]:
        """
        Return conversation context as Groq message dicts:
          1. The latest rolling summary (if one exists), as a system message
          2. The last `recent_limit` raw turns verbatim

        This gives the LLM long-term awareness + recent verbatim detail.
        """
        messages: list[dict] = []

        # 1. Latest summary
        cur = self._conn.execute(
            """
            SELECT content FROM conversations
            WHERE is_summary = 1
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        if row:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary so far]: {row['content']}",
            })

        # 2. Recent verbatim turns (non-summary only)
        cur = self._conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE is_summary = 0
            ORDER BY id DESC LIMIT ?
            """,
            (recent_limit,),
        )
        raw_turns = [{"role": r["role"], "content": r["content"]}
                     for r in reversed(cur.fetchall())]
        messages.extend(raw_turns)

        return messages

    def get_recent_raw(self, limit: int = RECENT_TURNS) -> list[dict[str, Any]]:
        """Return recent raw turns as plain dicts (for memory.py compatibility)."""
        cur = self._conn.execute(
            """
            SELECT role, content, timestamp FROM conversations
            WHERE is_summary = 0
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in reversed(cur.fetchall())]

    def _count_raw_turns(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE is_summary = 0"
        )
        return cur.fetchone()[0]

    # ── Summarisation ──────────────────────────────────────────────────────────

    def maybe_summarise(self) -> bool:
        """
        If raw turn count exceeds SUMMARY_THRESHOLD, compress old turns into
        a rolling summary and delete them. Returns True if summarisation ran.
        """
        total = self._count_raw_turns()
        if total <= SUMMARY_THRESHOLD:
            return False

        # Turns to summarise = everything beyond the RECENT_TURNS we keep
        keep = RECENT_TURNS
        compress_count = total - keep

        if compress_count <= 0:
            return False

        # Fetch oldest `compress_count` raw turns
        cur = self._conn.execute(
            """
            SELECT id, role, content FROM conversations
            WHERE is_summary = 0
            ORDER BY id ASC LIMIT ?
            """,
            (compress_count,),
        )
        old_turns = cur.fetchall()
        if not old_turns:
            return False

        ids_to_delete = [r["id"] for r in old_turns]

        # Load existing summary to incorporate
        cur = self._conn.execute(
            "SELECT content FROM conversations WHERE is_summary = 1 ORDER BY id DESC LIMIT 1"
        )
        existing_summary_row = cur.fetchone()
        existing_summary = existing_summary_row["content"] if existing_summary_row else ""

        # Build the text to summarise — cap each turn to 200 chars to stay within TPM
        turn_text = "\n".join(
            f"{r['role'].upper()}: {r['content'][:200]}" for r in old_turns
        )
        # Cap total input to 2000 chars to stay well within 6k TPM limit
        turn_text = turn_text[:2000]

        if existing_summary:
            user_content = (
                f"Previous summary:\n{existing_summary[:600]}\n\n"
                f"New turns to incorporate:\n{turn_text}"
            )
        else:
            user_content = f"Conversation turns to summarise:\n{turn_text}"

        try:
            response = self._groq.chat.completions.create(
                model=config.GROQ_MODEL_FAST,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            new_summary = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("[CSB] Summarisation API call failed: %s", exc)
            return False

        # Atomically: delete old raw turns + old summaries + insert new summary
        with self._conn:
            placeholders = ",".join("?" * len(ids_to_delete))
            self._conn.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            self._conn.execute(
                "DELETE FROM conversations WHERE is_summary = 1"
            )
            self._conn.execute(
                "INSERT INTO conversations (role, content, is_summary) VALUES ('system', ?, 1)",
                (new_summary,),
            )

        logger.info(
            "[CSB] Summarised %d turns into %d-char summary. %d raw turns remain.",
            compress_count, len(new_summary), keep,
        )
        return True

    def clear(self) -> None:
        """Delete all raw turns and summaries (facts preserved)."""
        with self._conn:
            self._conn.execute("DELETE FROM conversations")
