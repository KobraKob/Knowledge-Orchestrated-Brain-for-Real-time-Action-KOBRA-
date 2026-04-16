"""
proactive/scheduler.py — Cron-style scheduled task runner.

Usage:
    scheduler = TaskScheduler(speaker)
    scheduler.add("morning", "0 9 * * *", lambda: brain.process(
        "Give me a morning briefing: today's date, any events, and one motivational line."
    ))
    # Called every minute from ProactiveEngine._loop()
    scheduler.tick()
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ScheduledTask:
    def __init__(self, name: str, cron_expr: str, action_fn):
        try:
            from croniter import croniter
            self.cron = croniter(cron_expr, datetime.now())
        except ImportError:
            self.cron = None
            logger.warning("croniter not installed — scheduled task '%s' disabled.", name)
        self.name = name
        self.action = action_fn
        self.last_run: datetime | None = None
        self._next: datetime | None = self.cron.get_next(datetime) if self.cron else None

    def is_due(self) -> bool:
        if self._next is None:
            return False
        return datetime.now() >= self._next

    def advance(self) -> None:
        if self.cron:
            self._next = self.cron.get_next(datetime)


class TaskScheduler:
    def __init__(self, speaker):
        self._tasks: list[ScheduledTask] = []
        self._speaker = speaker

    def add(self, name: str, cron: str, action_fn) -> None:
        self._tasks.append(ScheduledTask(name, cron, action_fn))
        logger.info("[SCHEDULER] Registered: %s (%s)", name, cron)

    def tick(self) -> None:
        """Call every minute from ProactiveEngine._loop(). Runs due tasks."""
        now = datetime.now()
        for task in self._tasks:
            if not task.is_due():
                continue
            try:
                result = task.action()
                if result:
                    self._speaker.speak(result)
                task.last_run = now
                task.advance()
                logger.info("[SCHEDULER] Ran: %s", task.name)
            except Exception as exc:
                logger.error("[SCHEDULER] Task %s failed: %s", task.name, exc)
                task.advance()
