"""
routing_memory.py — Routing memory for KOBRA's Neural Router.

Stores (transcript, agent_used, outcome) tuples in SQLite.
Retrieves similar past successes as few-shot examples for the decomposition prompt.
Uses simple keyword overlap for similarity (no embeddings needed — fast).
"""

import sqlite3
import logging
import re
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_DB_PATH = "kobra_routing_memory.db"

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "me", "my",
    "can", "you", "i", "please", "hey", "kobra", "sir",
})


def _keywords(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class RoutingMemory:
    def __init__(self, db_path: str = _DB_PATH):
        self._db = Path(db_path)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                transcript  TEXT NOT NULL,
                agent       TEXT NOT NULL,
                instruction TEXT NOT NULL,
                outcome     TEXT NOT NULL  -- 'success' | 'failure' | 'corrected:<agent>'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_corrections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                transcript  TEXT NOT NULL,
                wrong_agent TEXT NOT NULL,
                right_agent TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def log_routing(
        self,
        transcript: str,
        agent: str,
        instruction: str,
        outcome: str = "success",
    ) -> None:
        """Log a routing decision and its outcome."""
        try:
            self._conn.execute(
                "INSERT INTO routing_log (timestamp, transcript, agent, instruction, outcome) VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), transcript, agent, instruction, outcome),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("[ROUTING_MEMORY] log_routing failed: %s", exc)

    def log_correction(self, transcript: str, wrong_agent: str, right_agent: str) -> None:
        """Log when a user corrects a routing decision."""
        try:
            self._conn.execute(
                "INSERT INTO routing_corrections (timestamp, transcript, wrong_agent, right_agent) VALUES (?,?,?,?)",
                (datetime.utcnow().isoformat(), transcript, wrong_agent, right_agent),
            )
            self._conn.commit()
            logger.info("[ROUTING_MEMORY] Correction logged: %s → %s", wrong_agent, right_agent)
        except Exception as exc:
            logger.warning("[ROUTING_MEMORY] log_correction failed: %s", exc)

    def get_few_shot_examples(self, transcript: str, limit: int = 3) -> str:
        """
        Return a string of few-shot routing examples similar to this transcript.
        Only returns SUCCESSFUL routings.
        """
        try:
            query_kws = _keywords(transcript)
            if not query_kws:
                return ""

            # Fetch recent successful routings
            rows = self._conn.execute(
                "SELECT transcript, agent, instruction FROM routing_log "
                "WHERE outcome = 'success' ORDER BY id DESC LIMIT 200"
            ).fetchall()

            if not rows:
                return ""

            # Score by keyword overlap
            scored = []
            for (t, agent, instr) in rows:
                row_kws = _keywords(t)
                overlap = len(query_kws & row_kws)
                if overlap > 0:
                    scored.append((overlap, t, agent, instr))

            scored.sort(key=lambda x: -x[0])
            top = scored[:limit]

            if not top:
                return ""

            lines = ["Past successful routings (few-shot examples):"]
            for (_, t, agent, instr) in top:
                lines.append(f'  User: "{t}" → agent: {agent}')
            return "\n".join(lines) + "\n\n"

        except Exception as exc:
            logger.warning("[ROUTING_MEMORY] get_few_shot_examples failed: %s", exc)
            return ""

    def get_agent_success_rates(self) -> dict[str, dict]:
        """Return success/failure counts per agent for diagnostics."""
        try:
            rows = self._conn.execute(
                "SELECT agent, outcome, COUNT(*) FROM routing_log GROUP BY agent, outcome"
            ).fetchall()
            stats: dict[str, dict] = {}
            for (agent, outcome, count) in rows:
                if agent not in stats:
                    stats[agent] = {"success": 0, "failure": 0}
                if outcome == "success":
                    stats[agent]["success"] += count
                else:
                    stats[agent]["failure"] += count
            return stats
        except Exception:
            return {}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
