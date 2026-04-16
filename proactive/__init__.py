"""proactive — background engine that surfaces alerts without being asked."""

from .engine import ProactiveEngine
from .checkers import (
    CalendarChecker,
    ProcessChecker,
    BehaviorChecker,
    BuildWatcherChecker,
    ClipboardMonitor,
    IdleChecker,
)
from .scheduler import TaskScheduler

__all__ = [
    "ProactiveEngine",
    "CalendarChecker",
    "ProcessChecker",
    "BehaviorChecker",
    "BuildWatcherChecker",
    "ClipboardMonitor",
    "IdleChecker",
    "TaskScheduler",
]
