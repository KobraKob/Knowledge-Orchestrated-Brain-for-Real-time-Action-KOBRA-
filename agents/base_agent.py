"""
agents/base_agent.py — Abstract base class for all KOBRA specialist agents.

Every agent:
  - Owns a specific subset of tools (OWNED_TOOLS)
  - Has a focused system prompt (SYSTEM_PROMPT)
  - Receives a Task, executes it via brain.process_scoped, returns a TaskResult
  - Checks the abort_flag before starting and honours cooperative cancellation
"""

import logging
import threading
import time
from abc import ABC, abstractmethod

from models import Task, TaskResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    AGENT_NAME: str = ""
    OWNED_TOOLS: list[str] = []
    SYSTEM_PROMPT: str = ""

    def __init__(self, brain, memory) -> None:
        self._brain = brain
        self._memory = memory

    # ── Public interface (called by task_queue.py) ─────────────────────────────

    def execute(self, task: Task, abort_flag: threading.Event) -> TaskResult:
        """Run the task and return a TaskResult. Never raises."""
        if abort_flag.is_set():
            logger.info("[%s] Skipped (aborted): %s", self.AGENT_NAME, task.id)
            return TaskResult(
                task_id=task.id,
                agent_name=self.AGENT_NAME,
                success=False,
                output="Task skipped — abort requested.",
                was_aborted=True,
            )

        start = time.perf_counter()
        logger.info("[%s] Starting task %s: %.80s", self.AGENT_NAME, task.id, task.instruction)

        try:
            output = self._run(task)
            success = True
        except Exception as exc:
            logger.exception("[%s] Task %s raised: %s", self.AGENT_NAME, task.id, exc)
            output = f"Error: {exc}"
            success = False

        duration = time.perf_counter() - start
        logger.info("[%s] Task %s done in %.2fs", self.AGENT_NAME, task.id, duration)

        return TaskResult(
            task_id=task.id,
            agent_name=self.AGENT_NAME,
            success=success,
            output=output,
            duration_seconds=duration,
        )

    # ── Subclass implements this ───────────────────────────────────────────────

    @abstractmethod
    def _run(self, task: Task) -> str:
        """Execute the task and return a result string."""
        ...

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_instruction(self, task: Task) -> str:
        """Merge task instruction with any injected dependency context."""
        if task.injected_context:
            return (
                f"{task.instruction}\n\n"
                f"Context from a previous step:\n{task.injected_context}"
            )
        return task.instruction
