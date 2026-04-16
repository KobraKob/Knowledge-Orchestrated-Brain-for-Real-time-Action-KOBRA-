"""
integrations/whatsapp.py — WhatsApp Web automation for KOBRA.

Uses Playwright with a persistent browser context (kobra_browser_session/).
First run: opens Chromium with a QR code — scan once, session saved forever.
Subsequent runs: session auto-restored, headless=True in config after first login.

Contact phone numbers must be in international format: +919876543210

All selectors stored in config.WHATSAPP_SELECTORS so UI changes are a 1-line fix.
"""

import logging
import time
import urllib.parse

import config
from contact_store import ContactStore, ContactNotFoundError
from credential_store import CredentialStore
from integrations.base_integration import BaseIntegration, IntegrationError

logger = logging.getLogger(__name__)

WHATSAPP_URL = "https://web.whatsapp.com"
_DEFAULT_SELECTORS = {
    "send_button":   '[data-testid="send"]',
    "msg_input":     '[data-testid="conversation-compose-box-input"]',
    "chat_list":     '[data-testid="chat-list"]',
    "qr_code":       '[data-testid="qrcode"]',
    "msg_container": '[data-testid="msg-container"]',
}


class WhatsAppIntegration(BaseIntegration):
    SERVICE_NAME = "whatsapp"

    def __init__(
        self,
        credential_store: CredentialStore,
        contact_store: ContactStore,
        page=None,
    ) -> None:
        self._creds_store = credential_store   # not used for WA — session = browser context
        self._contacts = contact_store
        self._page = page    # Playwright Page object — injected by BrowserAgent
        self._selectors = {**_DEFAULT_SELECTORS, **getattr(config, "WHATSAPP_SELECTORS", {})}

    def set_page(self, page) -> None:
        """Called by BrowserAgent after browser context is ready."""
        self._page = page

    def _sel(self, key: str) -> str:
        return self._selectors.get(key, _DEFAULT_SELECTORS.get(key, ""))

    # ── Auth ───────────────────────────────────────────────────────────────────

    def ensure_authenticated(self) -> bool:
        """
        Navigate to WhatsApp Web and check if logged in.
        If QR code is shown: wait up to 90s for the user to scan.
        Returns True when chat list is visible.
        """
        if self._page is None:
            logger.error("[WHATSAPP] No Playwright page set. BrowserAgent must inject it first.")
            return False

        try:
            self._page.goto(WHATSAPP_URL, timeout=30_000)

            # Already logged in?
            try:
                self._page.wait_for_selector(self._sel("chat_list"), timeout=10_000)
                logger.info("[WHATSAPP] Session active.")
                return True
            except Exception:
                pass

            # QR code shown — wait for user to scan
            try:
                self._page.wait_for_selector(self._sel("qr_code"), timeout=8_000)
                logger.info("[WHATSAPP] QR code detected — waiting for scan (90s).")
                # Wait for chat list to appear after scan
                self._page.wait_for_selector(self._sel("chat_list"), timeout=90_000)
                logger.info("[WHATSAPP] QR scan complete. Logged in.")
                return True
            except Exception:
                logger.error("[WHATSAPP] QR scan timed out or unexpected page state.")
                return False

        except Exception as exc:
            logger.error("[WHATSAPP] ensure_authenticated failed: %s", exc)
            return False

    # ── Actions ────────────────────────────────────────────────────────────────

    def send_message(self, to_name: str, message: str) -> str:
        """
        Send a WhatsApp message to a named contact.
        Uses the wa.me deep-link URL scheme — most reliable approach.
        """
        self._require_auth()

        contact = self._contacts.resolve(to_name)
        if contact is None:
            raise ContactNotFoundError(to_name)

        phone = contact.get("whatsapp") or contact.get("phone")
        if not phone:
            raise IntegrationError(
                f"I have {contact['name']} saved but no WhatsApp number or phone, sir. "
                f"Tell me their number and I'll save it."
            )

        # Normalize phone: remove spaces/dashes, ensure + prefix
        phone = phone.replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = "+" + phone

        encoded_msg = urllib.parse.quote(message)
        url = f"https://web.whatsapp.com/send?phone={phone}&text={encoded_msg}"

        try:
            self._page.goto(url, timeout=30_000)

            # Wait for message input to load (WhatsApp can be slow)
            self._page.wait_for_selector(self._sel("msg_input"), timeout=25_000)
            time.sleep(1)  # brief settle — WhatsApp pre-fills but needs a tick

            # Click send
            send_btn = self._page.query_selector(self._sel("send_button"))
            if send_btn:
                send_btn.click()
            else:
                # Fallback: press Enter
                self._page.keyboard.press("Enter")

            # Brief wait for send confirmation
            time.sleep(1.5)

            display = contact["name"]
            logger.info("[WHATSAPP] Message sent to %s (%s)", display, phone)
            return f"WhatsApp message sent to {display}."

        except IntegrationError:
            raise
        except Exception as exc:
            # Save a debug screenshot
            try:
                self._page.screenshot(path="whatsapp_error.png")
            except Exception:
                pass
            logger.error("[WHATSAPP] send_message failed: %s", exc)
            raise IntegrationError(f"WhatsApp send failed: {exc}") from exc

    def read_messages(self, from_name: str, count: int = 5) -> str:
        """
        Read the last N messages from a contact's chat.
        Returns a voice-friendly summary.
        """
        self._require_auth()

        contact = self._contacts.resolve(from_name)
        if contact is None:
            raise ContactNotFoundError(from_name)

        phone = contact.get("whatsapp") or contact.get("phone")
        if not phone:
            raise IntegrationError(f"No WhatsApp number for {contact['name']}, sir.")

        phone = phone.replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = "+" + phone

        url = f"https://web.whatsapp.com/send?phone={phone}"
        try:
            self._page.goto(url, timeout=30_000)
            self._page.wait_for_selector(self._sel("msg_input"), timeout=20_000)
            time.sleep(2)

            # Scrape message bubbles
            bubbles = self._page.query_selector_all('[data-testid="msg-container"]')
            texts = []
            for bubble in bubbles[-count:]:
                text = bubble.inner_text().strip()
                if text:
                    texts.append(text[:100])

            if not texts:
                return f"I couldn't read any messages from {contact['name']}, sir."

            display = contact["name"]
            return (
                f"Last {len(texts)} messages from {display}: "
                + " | ".join(texts)
            )

        except Exception as exc:
            logger.error("[WHATSAPP] read_messages failed: %s", exc)
            raise IntegrationError(f"Could not read WhatsApp messages: {exc}") from exc
