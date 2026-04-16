"""
proactive/engine.py — Background engine that runs checkers and speaks alerts.

Architecture:
  - Daemon thread runs _loop() every PROACTIVE_CHECK_INTERVAL seconds
  - Each checker returns a notification string or None
  - Cooldown dict prevents repeating the same alert every 30 seconds
  - Speaks via the injected speaker instance (decoupled)
"""

import logging
import threading
import time

import config
from .checkers import BaseChecker

logger = logging.getLogger(__name__)


class ProactiveEngine:
    """
    Runs background checkers and speaks alerts without being prompted.
    Wire up in main.py after speaker is ready:

        engine = ProactiveEngine(speaker=speaker, memory=memory)
        engine.register_checker(CalendarChecker(calendar_client))
        engine.register_checker(BehaviorChecker())
        engine.start()
    """

    def __init__(self, speaker, memory=None, calendar_client=None):
        self._speaker = speaker
        self._memory = memory
        self._calendar = calendar_client
        self._running = False
        self._thread: threading.Thread | None = None
        self._checkers: list[BaseChecker] = []
        self._cooldowns: dict[str, float] = {}
        self._scheduler = None  # injected after construction if needed

    def register_checker(self, checker: BaseChecker) -> None:
        self._checkers.append(checker)
        logger.info("[PROACTIVE] Registered checker: %s", checker.name)

    def set_scheduler(self, scheduler) -> None:
        self._scheduler = scheduler

    def start(self) -> None:
        if not config.PROACTIVE_ENABLED:
            logger.info("[PROACTIVE] Engine disabled via config.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="proactive")
        self._thread.start()
        logger.info("[PROACTIVE] Engine started (interval: %ds)", config.PROACTIVE_CHECK_INTERVAL)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._running:
            for checker in self._checkers:
                try:
                    notification = checker.check()
                    if notification and self._should_notify(checker.name, checker.cooldown_seconds):
                        logger.info("[PROACTIVE] %s: %s", checker.name, notification)
                        self._speaker.speak(notification)
                        self._set_cooldown(checker.name, checker.cooldown_seconds)
                except Exception as exc:
                    logger.debug("[PROACTIVE] Checker %s failed: %s", checker.name, exc)

            if self._scheduler:
                try:
                    self._scheduler.tick()
                except Exception as exc:
                    logger.debug("[PROACTIVE] Scheduler tick failed: %s", exc)

            time.sleep(config.PROACTIVE_CHECK_INTERVAL)

    def _should_notify(self, name: str, cooldown: int) -> bool:
        last = self._cooldowns.get(name, 0.0)
        return (time.time() - last) >= cooldown

    def _set_cooldown(self, name: str, cooldown: int) -> None:
        self._cooldowns[name] = time.time()
