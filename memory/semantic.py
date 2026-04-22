"""
memory/semantic.py — SemanticMemory for KOBRA v5.

Stores what is true about the user: facts, preferences, skills, identity.
Wraps existing memory.py facts table + learning.py semantic_memory table.

Facts are sticky — they persist until explicitly updated or contradicted.
access_count increments on retrieval — frequently used facts score higher.
"""

import logging
import re
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
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
            self.final_score = self.relevance_score * 0.7 + self.recency_score * 0.3


# Tech preference inference rules
_TECH_MAP = {
    "fastapi": ("preferred_web_framework", "FastAPI", "preference"),
    "flask":   ("preferred_web_framework", "Flask",   "preference"),
    "django":  ("preferred_web_framework", "Django",  "preference"),
    "react":   ("preferred_frontend",  "React",  "skill"),
    "vue":     ("preferred_frontend",  "Vue",    "skill"),
    "svelte":  ("preferred_frontend",  "Svelte", "skill"),
    "postgres": ("preferred_database", "PostgreSQL", "preference"),
    "mongodb":  ("preferred_database", "MongoDB",    "preference"),
    "pytest":  ("testing_framework", "pytest",   "skill"),
    "jest":    ("testing_framework", "Jest",     "skill"),
    "docker":  ("uses_docker",  "yes", "skill"),
    "vscode":  ("preferred_editor", "VS Code",  "preference"),
    "neovim":  ("preferred_editor", "Neovim",   "preference"),
    "python":  ("primary_language", "Python",   "skill"),
    "typescript": ("primary_language", "TypeScript", "skill"),
}


class SemanticMemory:
    """
    Unified semantic store. Reads from both:
      - existing memory.db facts table (brain.py memory.save_fact)
      - learning.db semantic_memory table (learning.py)
    Writes to both so nothing is lost during migration.
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                key          TEXT UNIQUE NOT NULL,
                value        TEXT NOT NULL,
                category     TEXT NOT NULL DEFAULT 'general',
                confidence   REAL DEFAULT 1.0,
                source       TEXT DEFAULT 'explicit',
                access_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                dimension    TEXT NOT NULL UNIQUE,
                value        TEXT NOT NULL,
                evidence     INTEGER DEFAULT 1,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Migration: add columns if missing
        for col, default in [("category", "'general'"), ("confidence", "1.0"),
                              ("source", "'explicit'"), ("access_count", "0"), ("last_updated", "CURRENT_TIMESTAMP")]:
            try:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {col.upper()} DEFAULT {default}")
            except Exception:
                pass
        self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_fact(self, key: str, value: str, category: str = "general",
                  confidence: float = 1.0, source: str = "explicit") -> None:
        now = datetime.utcnow().isoformat()
        try:
            self._conn.execute(
                """INSERT INTO facts (key, value, category, confidence, source, last_updated)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, confidence=excluded.confidence,
                     last_updated=excluded.last_updated""",
                (key.lower().strip(), value[:500], category, confidence, source, now),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[SEMANTIC] save_fact failed: %s", exc)

    def update_preference(self, dimension: str, value: str) -> None:
        now = datetime.utcnow().isoformat()
        try:
            self._conn.execute(
                """INSERT INTO preferences (dimension, value, evidence, last_updated)
                   VALUES (?,?,1,?)
                   ON CONFLICT(dimension) DO UPDATE SET
                     value=excluded.value,
                     evidence=evidence+1,
                     last_updated=excluded.last_updated""",
                (dimension, value, now),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[SEMANTIC] update_preference failed: %s", exc)

    def infer_and_update(self, episode: dict) -> None:
        """Lightweight rule-based inference from a conversation turn — no LLM."""
        transcript = (episode.get("transcript") or episode.get("content") or "").lower()
        if not transcript:
            return
        for kw, (key, value, cat) in _TECH_MAP.items():
            if kw in transcript:
                self.save_fact(key, value, cat, confidence=0.7, source="inferred")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all_facts(self) -> list[dict]:
        """Return all facts as plain dicts (memory.py compat)."""
        try:
            rows = self._conn.execute(
                "SELECT key, value, category FROM facts ORDER BY access_count DESC, last_updated DESC"
            ).fetchall()
            return [{"key": r["key"], "value": r["value"], "category": r["category"]} for r in rows]
        except Exception:
            return []

    def get_all_preferences(self) -> dict:
        try:
            rows = self._conn.execute("SELECT dimension, value FROM preferences").fetchall()
            return {r["dimension"]: r["value"] for r in rows}
        except Exception:
            return {}

    def get_known_contacts(self) -> set[str]:
        try:
            rows = self._conn.execute(
                "SELECT value FROM facts WHERE category='contact'"
            ).fetchall()
            return {r["value"].lower() for r in rows}
        except Exception:
            return set()

    def query(self, question: str, limit: int = 10) -> list[MemoryResult]:
        """Return semantic facts relevant to this question."""
        results = []
        try:
            # Always include identity/preference facts
            priority_rows = self._conn.execute(
                "SELECT key, value, category, last_updated FROM facts WHERE category IN ('identity','preference','project') ORDER BY access_count DESC LIMIT 5"
            ).fetchall()

            # Keyword-matched facts
            words = re.findall(r"\b\w{4,}\b", question.lower())
            kw_rows = []
            if words:
                ph = " OR ".join(["LOWER(key) LIKE ? OR LOWER(value) LIKE ?" for _ in words[:4]])
                params = [item for w in words[:4] for item in (f"%{w}%", f"%{w}%")]
                kw_rows = self._conn.execute(
                    f"SELECT key, value, category, last_updated FROM facts WHERE {ph} LIMIT ?",
                    params + [limit],
                ).fetchall()
                # Increment access count
                for r in kw_rows:
                    self._conn.execute("UPDATE facts SET access_count=access_count+1 WHERE key=?", (r["key"],))
                self._conn.commit()

            seen_keys = set()
            for row in list(priority_rows) + list(kw_rows):
                if row["key"] in seen_keys:
                    continue
                seen_keys.add(row["key"])
                ts = None
                try:
                    ts = datetime.fromisoformat(row["last_updated"])
                except Exception:
                    pass
                results.append(MemoryResult(
                    content=f"{row['key']}: {row['value']}",
                    source="semantic",
                    subsource="facts",
                    relevance_score=0.9 if row["category"] in ("identity", "preference") else 0.6,
                    recency_score=0.9,   # semantic facts don't decay
                    timestamp=ts,
                    metadata={"category": row["category"]},
                ))
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.debug("[SEMANTIC] query failed: %s", exc)
        return results

    def get_context_string(self, limit: int = 12) -> str:
        """Build a prompt-ready context block of semantic facts."""
        facts = self.get_all_facts()
        if not facts:
            return ""
        lines = ["User profile:"]
        for f in facts[:limit]:
            lines.append(f"  {f['key']}: {f['value']}")
        prefs = self.get_all_preferences()
        if prefs:
            lines.append("Preferences:")
            for dim, val in list(prefs.items())[:5]:
                lines.append(f"  {dim}: {val}")
        return "\n".join(lines)
