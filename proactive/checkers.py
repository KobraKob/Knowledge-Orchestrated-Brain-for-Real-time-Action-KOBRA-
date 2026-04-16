"""
proactive/checkers.py — Individual condition checkers for the ProactiveEngine.

Each checker monitors one condition and returns a spoken notification string
when that condition is met, or None if nothing to report.
"""

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BaseChecker:
    """Abstract base — implement check() to return a notification or None."""
    name: str = "base"
    cooldown_seconds: int = 300  # 5 minutes between notifications

    def check(self) -> str | None:
        raise NotImplementedError


class CalendarChecker(BaseChecker):
    """Alerts sir when a calendar event is starting soon."""
    name = "calendar"
    cooldown_seconds = 600

    def __init__(self, calendar_client=None):
        self._calendar = calendar_client

    def check(self) -> str | None:
        if not self._calendar:
            return None
        try:
            events = self._calendar.get_upcoming(minutes=15)
            for event in events:
                start = event.get("start")
                if not start:
                    continue
                # Ensure both datetimes are timezone-aware for comparison
                now = datetime.now(timezone.utc)
                if start.tzinfo is None:
                    # Naive start — treat as UTC to avoid TypeError
                    start = start.replace(tzinfo=timezone.utc)
                mins = int((start - now).total_seconds() / 60)
                if 1 <= mins <= 5:
                    return f"Sir, {event['title']} starts in {mins} minutes."
                elif 10 <= mins <= 15:
                    return f"Heads up sir — {event['title']} is in {mins} minutes."
        except Exception as exc:
            logger.debug("CalendarChecker error: %s", exc)
        return None


class ProcessChecker(BaseChecker):
    """Notifies when a watched process finishes."""
    name = "process"
    cooldown_seconds = 60

    def __init__(self, watched_processes: list[str] | None = None):
        self._watched = watched_processes or []
        self._running_procs: dict[str, float] = {}

    def check(self) -> str | None:
        try:
            import psutil
            current = {p.name() for p in psutil.process_iter(["name"])}
            for proc_name in self._watched:
                if proc_name in current and proc_name not in self._running_procs:
                    self._running_procs[proc_name] = time.time()
                elif proc_name not in current and proc_name in self._running_procs:
                    duration = int(time.time() - self._running_procs.pop(proc_name))
                    mins, secs = divmod(duration, 60)
                    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    return f"Sir, {proc_name} finished after {duration_str}."
        except Exception as exc:
            logger.debug("ProcessChecker error: %s", exc)
        return None


class BehaviorChecker(BaseChecker):
    """Nudges sir to take a break after extended work sessions."""
    name = "behavior"
    cooldown_seconds = 3600  # once per hour maximum

    def __init__(self):
        self._session_start = time.time()

    def check(self) -> str | None:
        elapsed_hours = (time.time() - self._session_start) / 3600
        if elapsed_hours >= 2.0:
            return "Sir, you've been working for over 2 hours. Might be worth a short break."
        hour = datetime.now().hour
        if hour == 13:
            return "Sir, it's 1 PM — usually around when you disappear for lunch."
        return None


class BuildWatcherChecker(BaseChecker):
    """Watches log files for error patterns and alerts on new failures."""
    name = "build_watcher"
    cooldown_seconds = 30

    def __init__(self, log_paths: list[str] | None = None):
        self._log_paths = log_paths or []
        self._last_sizes: dict[str, int] = {}

    def check(self) -> str | None:
        for path in self._log_paths:
            if not os.path.exists(path):
                continue
            try:
                size = os.path.getsize(path)
                if size == self._last_sizes.get(path, 0):
                    continue
                self._last_sizes[path] = size

                with open(path, "r", errors="ignore") as f:
                    lines = f.readlines()[-50:]
                content = "".join(lines).lower()

                if any(k in content for k in ("error", "failed", "exception", "traceback")):
                    filename = os.path.basename(path)
                    return f"Sir, {filename} shows errors. Want me to read them?"
            except Exception as exc:
                logger.debug("BuildWatcherChecker error for %s: %s", path, exc)
        return None


class IdleChecker(BaseChecker):
    """
    Speaks up after the user has been silent for ~2-3 minutes.
    Call update_activity() from the main loop after every user turn.
    """
    name = "idle"
    cooldown_seconds = 180  # at most once per 3 minutes

    IDLE_TIMEOUT = 150  # 2.5 minutes of silence before speaking

    _IDLE_MESSAGES = [
        "Sir, I appear to be experiencing an unusual silence. Still here if you need me.",
        "Everything quiet on your end, sir? I'm here when you're ready.",
        "Sir, still here — waiting with the patience of something that has no choice.",
        "The silence is deafening, sir. Anything I can help with?",
        "Sir, you've gone quiet. I haven't crashed — I checked.",
        "Still standing by, sir. Perfectly idle, perfectly fine.",
    ]

    def __init__(self):
        self._last_activity = time.time()
        self._message_index = 0

    def update_activity(self):
        """Call this after every user command to reset the idle clock."""
        self._last_activity = time.time()

    def check(self) -> str | None:
        if (time.time() - self._last_activity) >= self.IDLE_TIMEOUT:
            msg = self._IDLE_MESSAGES[self._message_index % len(self._IDLE_MESSAGES)]
            self._message_index += 1
            # Reset so the cooldown in ProactiveEngine also refreshes
            self._last_activity = time.time()
            return msg
        return None


class ClipboardMonitor(BaseChecker):
    """Offers to open URLs copied to clipboard."""
    name = "clipboard"
    cooldown_seconds = 60  # 10s was far too aggressive; 1 min between prompts

    def __init__(self):
        self._last_clip = ""

    def check(self) -> str | None:
        try:
            import pyperclip
            clip = pyperclip.paste()
            if clip == self._last_clip:
                return None
            self._last_clip = clip
            if clip.startswith("http") and len(clip) < 300:
                return "Sir, you copied a URL. Want me to open it?"
        except Exception:
            pass
        return None
