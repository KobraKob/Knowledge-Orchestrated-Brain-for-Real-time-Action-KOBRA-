"""
proactive/watcher.py — ContinuousWatcher for KOBRA v5.

Threshold-based monitoring (NOT interval polling):
  - Calendar: fires only when event is 5min or 15min away (not every 30s forever)
  - Email: fires only when new email arrives from known contact
  - Flow suppression: if user has been active in last 3 min, hold non-urgent alerts
  - Deduplication: same alert not repeated within 1 hour

Replaces the naive interval-based checkers from proactive/checkers.py.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import config

logger = logging.getLogger(__name__)

_FLOW_SUPPRESSION_SEC = getattr(config, "WATCHER_FLOW_SUPPRESSION_SEC", 180)
_CALENDAR_INTERVAL   = getattr(config, "WATCHER_CALENDAR_INTERVAL", 60)
_EMAIL_INTERVAL      = getattr(config, "WATCHER_EMAIL_INTERVAL", 300)
_PROJECT_INTERVAL    = getattr(config, "WATCHER_PROJECT_INTERVAL", 600)


class ContinuousWatcher:
    """
    Background watcher that monitors calendar, email, and project state.
    Notifies via speaker callback only when thresholds are crossed.
    Context-aware: suppresses non-urgent alerts during active use.
    """

    def __init__(
        self,
        speak_callback: Callable[[str], None],
        cal_client=None,
        email_client=None,
        semantic_memory=None,
    ) -> None:
        self._speak          = speak_callback
        self._calendar       = cal_client
        self._email          = email_client
        self._semantic       = semantic_memory
        self._last_activity  = datetime.now()
        self._notification_log: list[dict] = []
        self._last_email_check = datetime.now()
        self._known_event_ids: set[str] = set()
        self._stop_event     = threading.Event()
        self._threads: list[threading.Thread] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_activity(self) -> None:
        """Call from main.py every time user speaks — suppresses non-urgent alerts."""
        self._last_activity = datetime.now()

    def start(self) -> None:
        """Start all background watcher threads."""
        self._stop_event.clear()

        watchers = [
            ("calendar-watcher", self._calendar_loop, _CALENDAR_INTERVAL),
            ("email-watcher",    self._email_loop,    _EMAIL_INTERVAL),
        ]
        for name, target, interval in watchers:
            t = threading.Thread(target=target, args=(interval,), daemon=True, name=name)
            t.start()
            self._threads.append(t)
        logger.info("[WATCHER] Started %d watcher threads.", len(self._threads))

    def stop(self) -> None:
        """Stop all watcher threads."""
        self._stop_event.set()
        logger.info("[WATCHER] Stopped.")

    # ── Flow state ─────────────────────────────────────────────────────────────

    def _is_in_flow(self) -> bool:
        """True if user has been active recently — suppress non-urgent alerts."""
        return (datetime.now() - self._last_activity).seconds < _FLOW_SUPPRESSION_SEC

    def _already_notified(self, key: str, cooldown_seconds: int = 3600) -> bool:
        """Prevent duplicate notifications for the same event within cooldown period."""
        now = datetime.now()
        return any(
            n["key"] == key and (now - n["time"]).seconds < cooldown_seconds
            for n in self._notification_log
        )

    def _notify(self, message: str, key: str, urgent: bool = False) -> None:
        """Speak a notification if conditions are met."""
        if self._already_notified(key):
            return
        if self._is_in_flow() and not urgent:
            logger.debug("[WATCHER] Suppressed non-urgent alert (user in flow): %s", key)
            return
        self._notification_log.append({"key": key, "time": datetime.now()})
        # Prune old log entries
        cutoff = datetime.now() - timedelta(hours=2)
        self._notification_log = [n for n in self._notification_log if n["time"] > cutoff]
        logger.info("[WATCHER] Notifying: %s", message[:80])
        try:
            self._speak(message)
        except Exception as exc:
            logger.warning("[WATCHER] speak failed: %s", exc)

    # ── Calendar loop ──────────────────────────────────────────────────────────

    def _calendar_loop(self, interval: int) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_calendar()
            except Exception as exc:
                logger.debug("[WATCHER] calendar check failed: %s", exc)
            self._stop_event.wait(interval)

    def _check_calendar(self) -> None:
        if not self._calendar:
            return
        try:
            # Fetch events in next 20 minutes
            if hasattr(self._calendar, "get_upcoming"):
                events = self._calendar.get_upcoming(minutes=20) or []
            elif hasattr(self._calendar, "get_events_today"):
                events = self._calendar.get_events_today() or []
            else:
                return

            now = datetime.now()
            for event in events:
                try:
                    start = event.get("start") or event.get("time")
                    if isinstance(start, str):
                        # Try to parse
                        try:
                            start = datetime.fromisoformat(start)
                        except Exception:
                            continue
                    if not isinstance(start, datetime):
                        continue

                    mins = int((start - now).total_seconds() / 60)
                    title = event.get("title", event.get("summary", "Meeting"))
                    event_id = event.get("id", title)

                    if 4 <= mins <= 6:
                        self._notify(
                            f"Sir, {title} starts in 5 minutes.",
                            key=f"cal_{event_id}_5min",
                            urgent=True,
                        )
                    elif 13 <= mins <= 16:
                        self._notify(
                            f"Heads up sir — {title} in 15 minutes.",
                            key=f"cal_{event_id}_15min",
                        )
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("[WATCHER] _check_calendar error: %s", exc)

    # ── Email loop ─────────────────────────────────────────────────────────────

    def _email_loop(self, interval: int) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_email()
            except Exception as exc:
                logger.debug("[WATCHER] email check failed: %s", exc)
            self._stop_event.wait(interval)

    def _check_email(self) -> None:
        if not self._email:
            return
        try:
            known_contacts: set[str] = set()
            if self._semantic and hasattr(self._semantic, "get_known_contacts"):
                known_contacts = self._semantic.get_known_contacts()

            if hasattr(self._email, "get_new_since"):
                new_emails = self._email.get_new_since(self._last_email_check) or []
            elif hasattr(self._email, "get_unread"):
                new_emails = self._email.get_unread(max_results=3) or []
            else:
                return

            self._last_email_check = datetime.now()

            for email in new_emails:
                sender_name  = email.get("from_name", email.get("sender", "Someone"))
                sender_email = email.get("from_email", email.get("sender_email", ""))
                subject      = email.get("subject", "")
                email_id     = email.get("id", f"{sender_email}_{subject[:20]}")

                # Only notify for known contacts or urgent signals
                is_known = (
                    sender_name.lower() in known_contacts
                    or sender_email.lower() in known_contacts
                )
                is_urgent = any(w in subject.lower() for w in ("urgent", "asap", "deadline"))

                if is_known or is_urgent:
                    self._notify(
                        f"New {'urgent ' if is_urgent else ''}message from {sender_name}, sir. "
                        f"Subject: {subject[:60]}.",
                        key=f"email_{email_id}",
                        urgent=is_urgent,
                    )
        except Exception as exc:
            logger.debug("[WATCHER] _check_email error: %s", exc)
