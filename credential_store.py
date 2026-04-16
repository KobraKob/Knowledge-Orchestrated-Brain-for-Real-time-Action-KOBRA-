"""
credential_store.py — Encrypted local storage for OAuth tokens and API keys.

All secrets are encrypted with a machine-derived Fernet key (hostname + username hash).
This means: the file is unreadable if copied to another machine, but requires
no password prompt at startup — hands-free operation is preserved.

Usage:
    store = CredentialStore()
    store.save("gmail", {"token": "...", "refresh_token": "...", "expiry": "..."})
    data = store.load("gmail")          # dict or None
    store.is_valid("gmail")             # False if expired or missing
    store.delete("gmail")               # logout
"""

import hashlib
import json
import logging
import os
import socket
import sqlite3
from base64 import urlsafe_b64encode
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken

import config

logger = logging.getLogger(__name__)


class CredentialStore:
    def __init__(self) -> None:
        self._key = self._get_machine_key()
        self._fernet = Fernet(self._key)
        self._conn = sqlite3.connect(config.CREDENTIALS_DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("CredentialStore ready — DB: %s", config.CREDENTIALS_DB_PATH)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    service    TEXT UNIQUE NOT NULL,
                    token      TEXT NOT NULL,
                    expires_at TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );
            """)

    # ── Machine-derived encryption key ─────────────────────────────────────────

    def _get_machine_key(self) -> bytes:
        """
        Derive a 32-byte Fernet key from hostname + username.
        Never stored — re-derived on every run. Protects against copying
        the DB file to another machine.
        """
        try:
            identity = f"{socket.gethostname()}::{os.getlogin()}"
        except Exception:
            identity = "kobra-fallback-identity"
        digest = hashlib.sha256(identity.encode()).digest()
        return urlsafe_b64encode(digest)

    # ── Writes ─────────────────────────────────────────────────────────────────

    def save(self, service: str, token_data: dict) -> None:
        """Encrypt and upsert token_data for the given service."""
        raw = json.dumps(token_data).encode()
        encrypted = self._fernet.encrypt(raw).decode()

        # Extract expiry if present (Google tokens use 'expiry' as ISO string)
        expiry = token_data.get("expiry") or token_data.get("expires_at")

        with self._conn:
            self._conn.execute("""
                INSERT INTO credentials (service, token, expires_at, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(service) DO UPDATE SET
                    token      = excluded.token,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
            """, (service, encrypted, str(expiry) if expiry else None))
        logger.info("[CREDS] Saved credentials for: %s", service)

    def delete(self, service: str) -> None:
        """Remove stored credentials (logout / re-auth)."""
        with self._conn:
            self._conn.execute("DELETE FROM credentials WHERE service = ?", (service,))
        logger.info("[CREDS] Deleted credentials for: %s", service)

    # ── Reads ──────────────────────────────────────────────────────────────────

    def load(self, service: str) -> dict | None:
        """Decrypt and return stored token dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT token FROM credentials WHERE service = ?", (service,)
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            decrypted = self._fernet.decrypt(row["token"].encode())
            return json.loads(decrypted)
        except (InvalidToken, json.JSONDecodeError) as exc:
            logger.error("[CREDS] Decryption failed for %s: %s", service, exc)
            return None

    def is_valid(self, service: str) -> bool:
        """
        Return True if credentials exist and are not expired.
        Expired tokens can often be refreshed — callers handle that separately.
        """
        data = self.load(service)
        if data is None:
            return False

        # Check expiry if present
        expiry_str = data.get("expiry") or data.get("expires_at")
        if expiry_str:
            try:
                expiry = datetime.fromisoformat(str(expiry_str).replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) >= expiry:
                    logger.info("[CREDS] Token for %s is expired.", service)
                    return False
            except (ValueError, TypeError):
                pass  # Unparseable expiry — assume still valid, let API reject it

        return True

    def close(self) -> None:
        self._conn.close()
