"""
integrations/gmail.py — Gmail integration for KOBRA.

Uses the Google Gmail API v1 via google-api-python-client.
OAuth tokens are stored encrypted in CredentialStore.

Scopes:
  - https://www.googleapis.com/auth/gmail.send
  - https://www.googleapis.com/auth/gmail.readonly
  - https://www.googleapis.com/auth/gmail.modify   (for marking read)

First run: opens browser for Google OAuth consent screen.
Subsequent runs: auto-refreshes token using saved refresh_token.
"""

import base64
import logging
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import config
from contact_store import ContactStore, ContactNotFoundError
from credential_store import CredentialStore
from integrations.base_integration import BaseIntegration, IntegrationError

logger = logging.getLogger(__name__)

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",   # shared token with Calendar
]


class GmailIntegration(BaseIntegration):
    SERVICE_NAME = "gmail"

    def __init__(self, credential_store: CredentialStore, contact_store: ContactStore) -> None:
        self._creds_store = credential_store
        self._contacts = contact_store
        self._service = None     # lazy — built on first authenticated call
        self._creds = None

    # ── Auth ───────────────────────────────────────────────────────────────────

    def ensure_authenticated(self) -> bool:
        """Load saved token or run OAuth flow. Returns True if authenticated."""
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            logger.error("[GMAIL] google-api-python-client not installed. "
                         "Run: pip install google-api-python-client google-auth-oauthlib")
            return False

        creds = None
        token_data = self._creds_store.load(self.SERVICE_NAME)

        if token_data:
            try:
                creds = Credentials.from_authorized_user_info(token_data, GMAIL_SCOPES)
            except Exception as exc:
                logger.warning("[GMAIL] Stored token invalid: %s", exc)
                creds = None

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_creds(creds)
                logger.info("[GMAIL] Token refreshed.")
            except Exception as exc:
                logger.warning("[GMAIL] Token refresh failed: %s", exc)
                creds = None

        # Full OAuth flow if needed
        if not creds or not creds.valid:
            creds_path = getattr(config, "GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
            if not os.path.exists(creds_path):
                logger.error(
                    "[GMAIL] %s not found. Download it from Google Cloud Console "
                    "(APIs & Services → Credentials → OAuth 2.0 Client IDs → Download JSON).",
                    creds_path,
                )
                return False
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, GMAIL_SCOPES)
                creds = flow.run_local_server(port=0, open_browser=True)
                self._save_creds(creds)
                logger.info("[GMAIL] OAuth flow completed, token saved.")
            except Exception as exc:
                logger.error("[GMAIL] OAuth flow failed: %s", exc)
                return False

        self._creds = creds
        self._service = None   # reset so _get_service rebuilds
        return True

    def _save_creds(self, creds) -> None:
        self._creds_store.save(self.SERVICE_NAME, {
            "token":         creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri":     creds.token_uri,
            "client_id":     creds.client_id,
            "client_secret": creds.client_secret,
            "scopes":        list(creds.scopes or []),
            "expiry":        creds.expiry.isoformat() if creds.expiry else None,
        })

    def get_raw_credentials(self):
        """Return the google.oauth2.credentials.Credentials object. Used by Calendar."""
        self._require_auth()
        return self._creds

    def _get_service(self):
        if self._service is None:
            self._require_auth()
            from googleapiclient.discovery import build
            self._service = build("gmail", "v1", credentials=self._creds)
        return self._service

    # ── Actions ────────────────────────────────────────────────────────────────

    def send_email(self, to_name: str, subject: str, body: str) -> str:
        """
        Send an email to a named contact or raw email address.
        Resolves name via ContactStore, or uses email directly if valid email format.
        Returns voice-friendly confirmation string.
        """
        import re
        self._require_auth()

        # Check if to_name is a raw email address
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, to_name.strip()):
            # It's a raw email address
            to_email = to_name.strip()
            display_name = to_email
        else:
            # Try to resolve as contact name
            contact = self._contacts.resolve(to_name)
            if contact is None or not contact.get("email"):
                raise ContactNotFoundError(to_name)
            to_email = contact["email"]
            display_name = contact["name"]

        # Build the MIME message
        msg = MIMEMultipart("alternative")
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        try:
            svc = self._get_service()
            svc.users().messages().send(userId="me", body={"raw": raw}).execute()
            logger.info("[GMAIL] Email sent to %s (%s)", display_name, to_email)
            return f"Email sent to {display_name} at {to_email}."
        except Exception as exc:
            logger.error("[GMAIL] send_email failed: %s", exc)
            raise IntegrationError(f"Gmail send failed: {exc}") from exc

    def read_emails(self, count: int = 5, query: str = "") -> str:
        """
        Read recent emails. Returns a voice-friendly summary.
        query: optional Gmail search string, e.g. 'is:unread from:john'
        """
        self._require_auth()
        svc = self._get_service()

        try:
            q = query or "in:inbox"
            result = svc.users().messages().list(
                userId="me", q=q, maxResults=count
            ).execute()
            messages = result.get("messages", [])
            if not messages:
                return "Your inbox is empty, sir."

            summaries = []
            for m in messages[:count]:
                msg = svc.users().messages().get(
                    userId="me", id=m["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]
                ).execute()
                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                sender = _clean_sender(headers.get("From", "Unknown"))
                subject = headers.get("Subject", "(no subject)")
                snippet = msg.get("snippet", "")[:80]
                summaries.append(f"One from {sender} about {subject!r}: {snippet}")

            count_word = "email" if len(summaries) == 1 else "emails"
            intro = f"You have {len(summaries)} {count_word}, sir. "
            return intro + " ".join(summaries)

        except Exception as exc:
            logger.error("[GMAIL] read_emails failed: %s", exc)
            raise IntegrationError(f"Could not read emails: {exc}") from exc

    def reply_email(self, message_id: str, body: str) -> str:
        """Reply to an email thread by message ID."""
        self._require_auth()
        svc = self._get_service()

        try:
            original = svc.users().messages().get(
                userId="me", id=message_id, format="metadata",
                metadataHeaders=["Subject", "From", "Message-ID", "References"]
            ).execute()
            headers = {h["name"]: h["value"]
                       for h in original.get("payload", {}).get("headers", [])}
            thread_id = original.get("threadId", message_id)

            reply = MIMEText(body, "plain")
            reply["To"] = headers.get("From", "")
            reply["Subject"] = "Re: " + headers.get("Subject", "")
            reply["In-Reply-To"] = headers.get("Message-ID", "")
            reply["References"] = headers.get("References", "") + " " + headers.get("Message-ID", "")

            raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
            svc.users().messages().send(
                userId="me", body={"raw": raw, "threadId": thread_id}
            ).execute()
            return "Reply sent."
        except Exception as exc:
            raise IntegrationError(f"Reply failed: {exc}") from exc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_sender(raw: str) -> str:
    """Extract display name from 'John Smith <john@example.com>'."""
    if "<" in raw:
        return raw.split("<")[0].strip().strip('"')
    return raw.strip()
