"""
contact_store.py — Contact resolution for KOBRA.

Resolves natural language names ("John", "mom", "my boss") → structured contact
info (email, phone, WhatsApp number). Built on SQLite with alias support and
fuzzy fallback via difflib.

Usage:
    store = ContactStore()
    store.save_contact("John Smith", aliases=["john", "my colleague"],
                       email="john@example.com", whatsapp="+911234567890")
    contact = store.resolve("john")   # → {name, email, phone, whatsapp, ...}
    contact = store.resolve("johnn")  # fuzzy → same result
"""

import difflib
import logging
import sqlite3
from typing import Any

import config

logger = logging.getLogger(__name__)


class ContactNotFoundError(Exception):
    """Raised when a name cannot be resolved to a contact."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"No contact found for: {name!r}")


class ContactStore:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(config.CONTACTS_DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("ContactStore ready — DB: %s", config.CONTACTS_DB_PATH)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    email      TEXT,
                    phone      TEXT,
                    whatsapp   TEXT,
                    slack_id   TEXT,
                    notes      TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS contact_aliases (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id INTEGER REFERENCES contacts(id) ON DELETE CASCADE,
                    alias      TEXT NOT NULL COLLATE NOCASE,
                    UNIQUE(alias)
                );
            """)

    # ── Writes ─────────────────────────────────────────────────────────────────

    def save_contact(
        self,
        name: str,
        aliases: list[str] | None = None,
        email: str | None = None,
        phone: str | None = None,
        whatsapp: str | None = None,
        slack_id: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Insert or update a contact. Returns the contact_id.
        Automatically adds the lowercase name as an alias.
        """
        with self._conn:
            # Check if contact already exists by name (case-insensitive)
            cur = self._conn.execute(
                "SELECT id FROM contacts WHERE name = ? COLLATE NOCASE", (name,)
            )
            row = cur.fetchone()

            if row:
                contact_id = row["id"]
                # Update fields that are provided
                updates = []
                vals = []
                for field, val in [("email", email), ("phone", phone),
                                    ("whatsapp", whatsapp), ("slack_id", slack_id),
                                    ("notes", notes)]:
                    if val is not None:
                        updates.append(f"{field} = ?")
                        vals.append(val)
                if updates:
                    vals.append(contact_id)
                    self._conn.execute(
                        f"UPDATE contacts SET {', '.join(updates)} WHERE id = ?", vals
                    )
            else:
                cur = self._conn.execute(
                    """INSERT INTO contacts (name, email, phone, whatsapp, slack_id, notes)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (name, email, phone, whatsapp, slack_id, notes),
                )
                contact_id = cur.lastrowid

            # Always add lowercased name as alias + any extra aliases
            all_aliases = list({name.lower()} | {a.lower() for a in (aliases or [])})
            for alias in all_aliases:
                self._conn.execute(
                    """INSERT OR IGNORE INTO contact_aliases (contact_id, alias)
                       VALUES (?, ?)""",
                    (contact_id, alias),
                )

        logger.info("[CONTACTS] Saved: %s (id=%d)", name, contact_id)
        return contact_id

    def update_field(self, name: str, field: str, value: str) -> bool:
        """Update a single field for an existing contact. Returns True if found."""
        valid_fields = {"email", "phone", "whatsapp", "slack_id", "notes"}
        if field not in valid_fields:
            return False
        cur = self._conn.execute(
            f"UPDATE contacts SET {field} = ? WHERE name = ? COLLATE NOCASE",
            (value, name),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Reads ──────────────────────────────────────────────────────────────────

    def resolve(self, name: str) -> dict[str, Any] | None:
        """
        Resolve a name or alias to a contact dict.
        Strategy:
          1. Exact alias match (case-insensitive)
          2. Exact name match (case-insensitive)
          3. Fuzzy match on all aliases (cutoff 0.6)
        Returns None if nothing found.
        """
        name_lower = name.strip().lower()

        # 1. Exact alias match
        cur = self._conn.execute("""
            SELECT c.* FROM contacts c
            JOIN contact_aliases a ON a.contact_id = c.id
            WHERE a.alias = ? COLLATE NOCASE
            LIMIT 1
        """, (name_lower,))
        row = cur.fetchone()
        if row:
            return dict(row)

        # 2. Exact name match
        cur = self._conn.execute(
            "SELECT * FROM contacts WHERE name = ? COLLATE NOCASE LIMIT 1", (name,)
        )
        row = cur.fetchone()
        if row:
            return dict(row)

        # 3. Fuzzy match across all aliases
        cur = self._conn.execute(
            "SELECT alias, contact_id FROM contact_aliases"
        )
        aliases = [(r["alias"], r["contact_id"]) for r in cur.fetchall()]
        alias_strings = [a[0] for a in aliases]

        matches = difflib.get_close_matches(name_lower, alias_strings, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            contact_id = next(cid for alias, cid in aliases if alias == best)
            cur = self._conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
            row = cur.fetchone()
            if row:
                logger.info("[CONTACTS] Fuzzy resolved %r → %r", name, row["name"])
                return dict(row)

        logger.info("[CONTACTS] No contact found for: %r", name)
        return None

    def get_all(self) -> list[dict[str, Any]]:
        """Return all contacts."""
        cur = self._conn.execute("SELECT * FROM contacts ORDER BY name")
        return [dict(r) for r in cur.fetchall()]

    def delete(self, name: str) -> bool:
        """Delete a contact and all their aliases."""
        cur = self._conn.execute(
            "DELETE FROM contacts WHERE name = ? COLLATE NOCASE", (name,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def format_for_voice(self, contact: dict) -> str:
        """Format a contact dict as a natural speech string."""
        parts = [contact["name"]]
        if contact.get("email"):
            parts.append(f"email: {contact['email']}")
        if contact.get("phone"):
            parts.append(f"phone: {contact['phone']}")
        if contact.get("whatsapp"):
            parts.append(f"WhatsApp: {contact['whatsapp']}")
        return ", ".join(parts)

    def close(self) -> None:
        self._conn.close()
