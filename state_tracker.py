"""
state_tracker.py — Explicit state machine for KOBRA's task execution.

States:
  PENDING   → task queued, not started
  RUNNING   → task currently executing
  SUCCESS   → task completed successfully
  FAILED    → task failed (may transition to RETRYING)
  RETRYING  → task failed, attempting again
  ABANDONED → retries exhausted, giving up
  SKIPPED   → dependency not met, cannot run

Rules:
  - Each command invocation gets a fresh ExecutionState
  - Mid-task interrupts (new user command) → current state ABANDONED, new state created
  - State transitions are logged to ExecutionJournal
  - No implicit transitions — every state change goes through transition()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    SUCCESS   = "SUCCESS"
    FAILED    = "FAILED"
    RETRYING  = "RETRYING"
    ABANDONED = "ABANDONED"
    SKIPPED   = "SKIPPED"


# Valid state transitions
_VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.PENDING:   {TaskState.RUNNING, TaskState.SKIPPED},
    TaskState.RUNNING:   {TaskState.SUCCESS, TaskState.FAILED},
    TaskState.FAILED:    {TaskState.RETRYING, TaskState.ABANDONED},
    TaskState.RETRYING:  {TaskState.RUNNING},
    TaskState.SUCCESS:   set(),   # terminal
    TaskState.ABANDONED: set(),   # terminal
    TaskState.SKIPPED:   set(),   # terminal
}

MAX_RETRIES = 3


@dataclass
class StepRecord:
    task_id:    str
    agent_name: str
    instruction: str
    state:      TaskState = TaskState.PENDING
    retries:    int       = 0
    output:     str       = ""
    error:      str       = ""
    started_at: float     = 0.0
    ended_at:   float     = 0.0
    retry_hint: str       = ""

    @property
    def duration(self) -> float:
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return 0.0


@dataclass
class ExecutionState:
    """
    Working memory for a single command invocation.
    Holds all step records and transition logic.
    Created fresh for every user command.
    """
    command: str
    steps:   dict[str, StepRecord] = field(default_factory=dict)
    created_at: float              = field(default_factory=time.time)
    interrupted: bool              = False    # True if a new command preempted this one
    _lock: threading.Lock          = field(default_factory=threading.Lock, repr=False)

    def add_step(self, task_id: str, agent_name: str, instruction: str) -> StepRecord:
        record = StepRecord(task_id=task_id, agent_name=agent_name, instruction=instruction)
        with self._lock:
            self.steps[task_id] = record
        return record

    def transition(self, task_id: str, new_state: TaskState, output: str = "", error: str = "", retry_hint: str = "") -> bool:
        """
        Attempt a state transition. Returns True if successful, False if invalid.
        """
        with self._lock:
            record = self.steps.get(task_id)
            if not record:
                logger.warning("[STATE] Unknown task_id: %s", task_id)
                return False

            current = record.state
            if new_state not in _VALID_TRANSITIONS.get(current, set()):
                logger.warning(
                    "[STATE] Invalid transition %s → %s for task %s",
                    current.value, new_state.value, task_id,
                )
                return False

            # Apply transition
            old_state = record.state
            record.state = new_state
            if output:
                record.output = output
            if error:
                record.error = error
            if retry_hint:
                record.retry_hint = retry_hint

            # Track timing
            if new_state == TaskState.RUNNING:
                record.started_at = time.time()
            elif new_state in (TaskState.SUCCESS, TaskState.FAILED, TaskState.ABANDONED, TaskState.SKIPPED):
                record.ended_at = time.time()
            elif new_state == TaskState.RETRYING:
                record.retries += 1
                record.ended_at = time.time()

            logger.info("[STATE] %s: %s → %s (retries=%d)", task_id, old_state.value, new_state.value, record.retries)
            return True

    def can_retry(self, task_id: str) -> bool:
        record = self.steps.get(task_id)
        return record is not None and record.retries < MAX_RETRIES

    def can_run(self, task_id: str, depends_on: list[str]) -> bool:
        """True if all dependencies are in SUCCESS state."""
        for dep_id in depends_on:
            dep = self.steps.get(dep_id)
            if dep is None or dep.state != TaskState.SUCCESS:
                return False
        return True

    def interrupt(self) -> None:
        """Called when a new user command preempts this execution."""
        with self._lock:
            self.interrupted = True
            for task_id, record in self.steps.items():
                if record.state in (TaskState.PENDING, TaskState.RUNNING, TaskState.RETRYING):
                    record.state = TaskState.ABANDONED
                    record.ended_at = time.time()
                    record.error = "Interrupted by new user command."
        logger.info("[STATE] Execution interrupted — all in-flight tasks ABANDONED.")

    def is_partial_success(self) -> bool:
        """True if some steps succeeded and some failed/abandoned."""
        states = {r.state for r in self.steps.values()}
        return TaskState.SUCCESS in states and (TaskState.FAILED in states or TaskState.ABANDONED in states)

    def get_partial_state_report(self) -> str:
        """Human-readable report of partial execution state."""
        lines = []
        for tid, record in self.steps.items():
            emoji = {"SUCCESS": "✓", "FAILED": "✗", "ABANDONED": "⚠", "SKIPPED": "–", "RUNNING": "…"}.get(record.state.value, "?")
            lines.append(f"{emoji} {record.agent_name}: {record.state.value}")
            if record.error:
                lines.append(f"  Error: {record.error[:80]}")
        return "\n".join(lines)

    def summary(self) -> dict:
        total = len(self.steps)
        by_state: dict[str, int] = {}
        for r in self.steps.values():
            by_state[r.state.value] = by_state.get(r.state.value, 0) + 1
        return {"total": total, "by_state": by_state, "interrupted": self.interrupted}


class StateTracker:
    """
    Manages ExecutionState lifecycle.
    Holds at most one active state per KOBRA session.
    When a new command arrives, the current state is interrupted and archived.
    """

    def __init__(self) -> None:
        self._current: ExecutionState | None = None
        self._history: list[ExecutionState] = []   # last 10 states
        self._lock = threading.Lock()

    def new_execution(self, command: str) -> ExecutionState:
        """Start a new execution, interrupting any in-flight one."""
        with self._lock:
            if self._current is not None:
                if not self._current.interrupted:
                    self._current.interrupt()
                self._history.append(self._current)
                if len(self._history) > 10:
                    self._history.pop(0)
            self._current = ExecutionState(command=command)
            return self._current

    def current(self) -> ExecutionState | None:
        return self._current

    def last_completed(self) -> ExecutionState | None:
        """Return the most recent completed (non-current) execution."""
        return self._history[-1] if self._history else None

    def get_partial_state_warning(self) -> str | None:
        """
        If the last execution was partial (some steps done, some not),
        return a warning string for the user.
        """
        last = self.last_completed()
        if last and last.is_partial_success():
            return (
                f"Warning: last command partially completed.\n"
                f"{last.get_partial_state_report()}\n"
                "Some actions may need to be undone manually."
            )
        return None
