"""
learning.py — KOBRA's learning system. Makes KOBRA feel alive over time.

Tracks:
  1. User vocabulary: names, project names, preferences extracted from transcripts
  2. Response style: which responses the user cuts off vs listens to fully
  3. Proactive patterns: time-of-day habits (e.g. asks weather every morning)
  4. Tool success rates: which agents solve which task types reliably
  5. Routing corrections: "no, use X for that" — adjusts future routing

All data stored in SQLite. Injected into brain.py prompts as personalization context.
"""

import sqlite3
import logging
import re
import json
from pathlib import Path
from datetime import datetime, date
from collections import Counter

logger = logging.getLogger(__name__)

_DB_PATH = "kobra_learning.db"

# Patterns to extract vocabulary from user speech
_NAME_PATTERNS = [
    r"\bmy (?:project|app|site|tool|script|bot|system|program) (?:called |named |is )?['\"]?([A-Z][a-zA-Z0-9_\- ]+)['\"]?",
    r"\bremember (?:that )?(?:my |I )?(?:name is |I'm called )?([A-Z][a-zA-Z ]+)",
    r"\bI(?:'m| am) ([A-Z][a-zA-Z ]+)",
    r"\bcall (?:it|this|the (?:project|app)) ['\"]?([A-Z][a-zA-Z0-9_\- ]+)['\"]?",
]

_PREFERENCE_PATTERNS = [
    r"\bI (?:always |usually |prefer to |like to |tend to )(.{5,40})",
    r"\bmy (?:favorite|preferred|usual) (.{3,30}) is ([a-zA-Z0-9 ]+)",
    r"\bI (?:use|work with|code in) ([a-zA-Z0-9\+\# ]+) (?:for|as my|as the)",
]


class LearningSystem:
    def __init__(self, db_path: str = _DB_PATH):
        self._db = Path(db_path)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._init_db()
        self._vocab_cache: dict[str, str] | None = None

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_vocabulary (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                term      TEXT NOT NULL UNIQUE,
                category  TEXT NOT NULL,  -- 'project_name' | 'person' | 'preference' | 'tool'
                value     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS response_feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                transcript   TEXT NOT NULL,
                agent        TEXT NOT NULL,
                response_len INTEGER NOT NULL,
                was_cut_off  INTEGER NOT NULL DEFAULT 0,  -- 1 if user interrupted
                rating       TEXT DEFAULT 'neutral'  -- 'good' | 'bad' | 'neutral'
            );

            CREATE TABLE IF NOT EXISTS usage_patterns (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                hour      INTEGER NOT NULL,
                weekday   INTEGER NOT NULL,
                agent     TEXT NOT NULL,
                topic     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS routing_corrections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                transcript  TEXT NOT NULL,
                wrong_agent TEXT NOT NULL,
                right_agent TEXT NOT NULL,
                weight      INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS tool_outcomes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent     TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success   INTEGER NOT NULL,
                duration  REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS semantic_memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                updated   TEXT NOT NULL,
                key       TEXT NOT NULL UNIQUE,   -- e.g. "preferred_framework"
                value     TEXT NOT NULL,          -- e.g. "FastAPI"
                category  TEXT NOT NULL,          -- 'preference' | 'identity' | 'skill' | 'project' | 'contact'
                confidence REAL NOT NULL DEFAULT 1.0,
                source    TEXT NOT NULL DEFAULT 'explicit'  -- 'explicit' | 'inferred'
            );

            CREATE TABLE IF NOT EXISTS episodic_memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                expires   TEXT NOT NULL,          -- 7 days from creation
                summary   TEXT NOT NULL,          -- "You ran pytest and it failed due to missing import"
                agent     TEXT NOT NULL,
                keywords  TEXT NOT NULL,          -- comma-separated for fast retrieval
                outcome   TEXT NOT NULL DEFAULT 'unknown'  -- 'success' | 'failure' | 'unknown'
            );
        """)
        self._conn.commit()

    # ── Vocabulary learning ────────────────────────────────────────────────────

    def extract_vocabulary(self, transcript: str) -> None:
        """Extract and store names, project names, preferences from transcript."""
        now = datetime.utcnow().isoformat()

        for pattern in _NAME_PATTERNS:
            for match in re.finditer(pattern, transcript, re.IGNORECASE):
                term = match.group(1).strip()
                if 2 < len(term) < 50:
                    self._upsert_vocab(now, term, "project_name", term)

        for pattern in _PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, transcript, re.IGNORECASE):
                term = match.group(1).strip()
                if 2 < len(term) < 60:
                    self._upsert_vocab(now, term, "preference", term)

    def _upsert_vocab(self, timestamp: str, term: str, category: str, value: str) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO user_vocabulary (timestamp, term, category, value) VALUES (?,?,?,?)",
                (timestamp, term.lower(), category, value),
            )
            self._conn.commit()
            self._vocab_cache = None  # invalidate cache
        except Exception as exc:
            logger.debug("[LEARNING] vocab upsert failed: %s", exc)

    def store_fact(self, term: str, category: str, value: str) -> None:
        """Explicitly store a user-provided fact."""
        self._upsert_vocab(datetime.utcnow().isoformat(), term, category, value)

    def get_personalization_context(self) -> str:
        """
        Build a personalization context string to inject into brain prompts.
        Returns empty string if nothing learned yet.
        """
        if self._vocab_cache is None:
            self._rebuild_vocab_cache()

        if not self._vocab_cache:
            return ""

        lines = ["User context (learned from past interactions):"]
        for term, value in list(self._vocab_cache.items())[:15]:  # cap at 15
            lines.append(f"  - {value}")
        return "\n".join(lines) + "\n\n"

    def _rebuild_vocab_cache(self) -> None:
        try:
            rows = self._conn.execute(
                "SELECT term, value FROM user_vocabulary ORDER BY id DESC LIMIT 30"
            ).fetchall()
            self._vocab_cache = {r[0]: r[1] for r in rows}
        except Exception:
            self._vocab_cache = {}

    # ── Usage pattern tracking ─────────────────────────────────────────────────

    def log_usage(self, agent: str, topic: str) -> None:
        """Log what the user asked for, when."""
        now = datetime.utcnow()
        try:
            self._conn.execute(
                "INSERT INTO usage_patterns (timestamp, hour, weekday, agent, topic) VALUES (?,?,?,?,?)",
                (now.isoformat(), now.hour, now.weekday(), agent, topic[:100]),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[LEARNING] log_usage failed: %s", exc)

    def get_proactive_suggestions(self) -> list[str]:
        """
        Detect habitual patterns. Returns list of proactive suggestion strings.
        Example: "User asks weather at 8am on weekdays" → suggest weather at 8am.
        """
        suggestions = []
        try:
            now = datetime.utcnow()
            # Check if user habitually does something at this hour
            rows = self._conn.execute(
                """SELECT topic, COUNT(*) as cnt FROM usage_patterns
                   WHERE hour = ? AND weekday = ? AND agent != 'conversation'
                   GROUP BY topic HAVING cnt >= 3
                   ORDER BY cnt DESC LIMIT 3""",
                (now.hour, now.weekday()),
            ).fetchall()
            for (topic, cnt) in rows:
                suggestions.append(f"You usually ask about {topic} at this time.")
        except Exception:
            pass
        return suggestions

    # ── Response feedback ──────────────────────────────────────────────────────

    def log_response(
        self,
        transcript: str,
        agent: str,
        response_len: int,
        was_cut_off: bool = False,
    ) -> None:
        """Log a response and whether the user cut it off (interrupt signal)."""
        try:
            self._conn.execute(
                "INSERT INTO response_feedback (timestamp, transcript, agent, response_len, was_cut_off) VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), transcript[:200], agent, response_len, int(was_cut_off)),
            )
            self._conn.commit()
        except Exception:
            pass

    def get_preferred_response_length(self, agent: str) -> int:
        """
        Return the average response length that WASN'T cut off for this agent.
        Returns 0 if not enough data.
        """
        try:
            row = self._conn.execute(
                """SELECT AVG(response_len) FROM response_feedback
                   WHERE agent = ? AND was_cut_off = 0 AND response_len > 0""",
                (agent,),
            ).fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception:
            return 0

    # ── Tool outcome tracking ──────────────────────────────────────────────────

    def log_tool_outcome(self, agent: str, task_type: str, success: bool, duration: float = 0.0) -> None:
        """Log whether an agent succeeded at a task type."""
        try:
            self._conn.execute(
                "INSERT INTO tool_outcomes (timestamp, agent, task_type, success, duration) VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), agent, task_type[:50], int(success), duration),
            )
            self._conn.commit()
        except Exception:
            pass

    def get_best_agent_for_task(self, task_type: str, candidates: list[str]) -> str | None:
        """
        Return the agent from candidates with the highest success rate for this task type.
        Returns None if not enough data.
        """
        try:
            scores = {}
            for agent in candidates:
                rows = self._conn.execute(
                    "SELECT success, COUNT(*) FROM tool_outcomes WHERE agent=? AND task_type=? GROUP BY success",
                    (agent, task_type),
                ).fetchall()
                total = sum(r[1] for r in rows)
                if total >= 3:  # need at least 3 samples
                    successes = sum(r[1] for r in rows if r[0] == 1)
                    scores[agent] = successes / total
            if scores:
                return max(scores, key=scores.__getitem__)
        except Exception:
            pass
        return None

    # ── Routing corrections ────────────────────────────────────────────────────

    def log_routing_correction(self, transcript: str, wrong_agent: str, right_agent: str) -> None:
        """User explicitly corrected a routing decision."""
        try:
            self._conn.execute(
                "INSERT INTO routing_corrections (timestamp, transcript, wrong_agent, right_agent) VALUES (?,?,?,?)",
                (datetime.utcnow().isoformat(), transcript[:200], wrong_agent, right_agent),
            )
            self._conn.commit()
            logger.info("[LEARNING] Routing correction: %s → %s", wrong_agent, right_agent)
        except Exception:
            pass

    def get_routing_correction_context(self) -> str:
        """
        Return recent routing corrections as a few-shot hint string for the decomposer.
        """
        try:
            rows = self._conn.execute(
                """SELECT transcript, wrong_agent, right_agent FROM routing_corrections
                   ORDER BY id DESC LIMIT 5"""
            ).fetchall()
            if not rows:
                return ""
            lines = ["User routing corrections (learn from these mistakes):"]
            for (t, wrong, right) in rows:
                lines.append(f'  "{t}" should use {right}, NOT {wrong}')
            return "\n".join(lines) + "\n\n"
        except Exception:
            return ""

    # ── Semantic memory ────────────────────────────────────────────────────────

    def store_semantic(self, key: str, value: str, category: str, confidence: float = 1.0, source: str = "explicit") -> None:
        """Store or update a semantic fact about the user."""
        now = datetime.utcnow().isoformat()
        try:
            self._conn.execute("""
                INSERT INTO semantic_memory (timestamp, updated, key, value, category, confidence, source)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated, confidence=excluded.confidence
            """, (now, now, key, value, category, confidence, source))
            self._conn.commit()
            self._vocab_cache = None
        except Exception as exc:
            logger.debug("[LEARNING] store_semantic failed: %s", exc)

    def get_semantic_context(self, limit: int = 12) -> str:
        """Return semantic facts as a context string for prompt injection."""
        try:
            rows = self._conn.execute(
                "SELECT key, value, category FROM semantic_memory ORDER BY confidence DESC, updated DESC LIMIT ?",
                (limit,)
            ).fetchall()
            if not rows:
                return ""
            lines = ["User profile (semantic memory):"]
            for (key, value, cat) in rows:
                lines.append(f"  [{cat}] {key}: {value}")
            return "\n".join(lines) + "\n"
        except Exception:
            return ""

    # ── Episodic memory ────────────────────────────────────────────────────────

    def store_episode(self, summary: str, agent: str, keywords: list[str], outcome: str = "unknown") -> None:
        """Store an episodic memory (expires in 7 days)."""
        from datetime import timedelta
        now = datetime.utcnow()
        expires = (now + timedelta(days=7)).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO episodic_memory (timestamp, expires, summary, agent, keywords, outcome) VALUES (?,?,?,?,?,?)",
                (now.isoformat(), expires, summary[:500], agent, ",".join(keywords[:10]), outcome)
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[LEARNING] store_episode failed: %s", exc)

    def get_relevant_episodes(self, query: str, limit: int = 3) -> str:
        """Return recent episodic memories relevant to this query."""
        try:
            now = datetime.utcnow().isoformat()
            # Get non-expired episodes
            rows = self._conn.execute(
                "SELECT summary, agent, outcome, timestamp FROM episodic_memory WHERE expires > ? ORDER BY id DESC LIMIT 50",
                (now,)
            ).fetchall()
            if not rows:
                return ""

            # Score by keyword overlap
            query_words = set(re.findall(r"\b\w+\b", query.lower()))
            scored = []
            for (summary, agent, outcome, ts) in rows:
                ep_words = set(re.findall(r"\b\w+\b", summary.lower()))
                overlap = len(query_words & ep_words)
                if overlap > 0:
                    scored.append((overlap, summary, agent, outcome, ts))

            scored.sort(key=lambda x: -x[0])
            top = scored[:limit]

            if not top:
                return ""

            lines = ["Relevant past events (episodic memory):"]
            for (_, summary, agent, outcome, ts) in top:
                date_str = ts[:10] if ts else "recently"
                lines.append(f"  [{date_str}] {summary} (outcome: {outcome})")
            return "\n".join(lines) + "\n"
        except Exception:
            return ""

    def cleanup_expired_episodes(self) -> None:
        """Remove expired episodic memories."""
        try:
            now = datetime.utcnow().isoformat()
            self._conn.execute("DELETE FROM episodic_memory WHERE expires < ?", (now,))
            self._conn.commit()
        except Exception:
            pass

    def infer_semantic_from_episode(self, transcript: str, agent: str, outcome: str) -> None:
        """
        Infer semantic facts from repeated episodic patterns.
        E.g., if user always uses 'pytest' → infer skill: Python testing.
        """
        t = transcript.lower()

        # Infer tech preferences
        tech_map = {
            "fastapi": ("preferred_web_framework", "FastAPI", "preference"),
            "flask": ("preferred_web_framework", "Flask", "preference"),
            "react": ("preferred_frontend", "React", "skill"),
            "vue": ("preferred_frontend", "Vue", "skill"),
            "postgres": ("preferred_database", "PostgreSQL", "preference"),
            "mongodb": ("preferred_database", "MongoDB", "preference"),
            "pytest": ("testing_framework", "pytest", "skill"),
            "docker": ("uses_docker", "yes", "skill"),
            "vscode": ("preferred_editor", "VS Code", "preference"),
        }
        for kw, (key, value, cat) in tech_map.items():
            if kw in t:
                self.store_semantic(key, value, cat, confidence=0.7, source="inferred")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
