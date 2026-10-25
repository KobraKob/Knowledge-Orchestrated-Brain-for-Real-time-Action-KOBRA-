"""
Verified task execution for KOBRA.

This module is an opt-in execution path around the existing Task/TaskResult and
agent interfaces. It does not replace TaskQueue. Unlike a successful agent
return, a task is only marked VERIFIED when a caller-supplied postcondition
verifier confirms the result.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Any

from models import Task, TaskResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    reason: str
    # Retries are opt-in. A failed postcondition may follow a side effect, so
    # retrying blindly could duplicate an email, payment, or other action.
    retryable: bool = False


@dataclass
class VerifiedStep:
    task_id: str
    agent_name: str
    instruction: str
    status: str = "PENDING"
    attempts: int = 0
    output: str = ""
    verification_reason: str = ""
    error: str = ""
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)


@dataclass
class ExecutionReport:
    execution_id: str
    objective: str
    status: str
    steps: list[VerifiedStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def verified_count(self) -> int:
        return sum(step.status == "VERIFIED" for step in self.steps)

    @property
    def success(self) -> bool:
        return bool(self.steps) and all(step.status == "VERIFIED" for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "objective": self.objective,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "verified_count": self.verified_count,
            "total_steps": len(self.steps),
            "success": self.success,
            "steps": [
                {**asdict(step), "duration_seconds": step.duration_seconds}
                for step in self.steps
            ],
        }


Verifier = Callable[[Task, TaskResult], VerificationResult]


class VerifiedTaskExecutor:
    """Run tasks in dependency order and verify postconditions before continuing.

    Agents must implement the existing `execute(task, abort_flag) -> TaskResult`
    interface. The verifier must inspect a real postcondition (for example,
    checking that a file exists and contains expected content), not merely
    repeat the agent's success message.
    """

    def __init__(
        self,
        verifier: Verifier,
        *,
        max_attempts: int = 2,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not callable(verifier):
            raise TypeError("verifier must be callable")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._checkpoint_lock = threading.Lock()

    def execute(
        self,
        objective: str,
        tasks: list[Task],
        agents: dict[str, Any],
        abort_flag: threading.Event | None = None,
    ) -> ExecutionReport:
        """Execute a task graph. Downstream tasks run only after verified deps."""
        abort_flag = abort_flag or threading.Event()
        self._validate_graph(tasks)
        execution_id = f"kobra-{int(time.time() * 1000)}"
        report = ExecutionReport(
            execution_id=execution_id,
            objective=objective,
            status="RUNNING",
            steps=[
                VerifiedStep(t.id, t.agent_name, t.instruction)
                for t in tasks
            ],
        )
        step_by_id = {step.task_id: step for step in report.steps}
        task_by_id = {task.id: task for task in tasks}
        outputs: dict[str, str] = {}
        pending = list(tasks)

        self._checkpoint(report)
        while pending:
            if abort_flag.is_set():
                for task in pending:
                    step = step_by_id[task.id]
                    step.status = "ABORTED"
                    step.error = "Execution aborted before this step started."
                break

            ready = [
                task for task in pending
                if all(step_by_id[dep].status in {"VERIFIED", "FAILED", "SKIPPED", "UNVERIFIED"}
                       for dep in task.depends_on)
            ]
            if not ready:
                # Graph validation catches cycles; no ready node means a
                # prerequisite did not reach a terminal state.
                for task in pending:
                    step = step_by_id[task.id]
                    step.status = "SKIPPED"
                    step.error = "Dependency could not reach a terminal state."
                break

            for task in ready:
                pending.remove(task)
                step = step_by_id[task.id]
                failed_dependencies = [
                    dep for dep in task.depends_on
                    if step_by_id[dep].status != "VERIFIED"
                ]
                if failed_dependencies:
                    step.status = "SKIPPED"
                    step.error = "Unverified dependencies: " + ", ".join(failed_dependencies)
                    self._checkpoint(report)
                    continue

                agent = agents.get(task.agent_name)
                if agent is None:
                    step.status = "FAILED"
                    step.error = f"No agent registered for '{task.agent_name}'."
                    step.ended_at = time.time()
                    self._checkpoint(report)
                    continue

                step.status = "RUNNING"
                step.started_at = time.time()
                context = "\n".join(
                    f"{dep}: {outputs[dep]}" for dep in task.depends_on if dep in outputs
                )
                execution_task = Task(
                    id=task.id,
                    agent_name=task.agent_name,
                    instruction=task.instruction,
                    can_parallelize=False,
                    depends_on=list(task.depends_on),
                    injected_context=context,
                )
                self._checkpoint(report)

                for attempt in range(1, self.max_attempts + 1):
                    step.attempts = attempt
                    if abort_flag.is_set():
                        step.status = "ABORTED"
                        step.error = "Execution aborted."
                        break
                    try:
                        result = agent.execute(execution_task, abort_flag)
                        if not isinstance(result, TaskResult):
                            raise TypeError("Agent execute() must return TaskResult")
                        step.output = str(result.output or "")
                        if result.was_aborted or abort_flag.is_set():
                            step.status = "ABORTED"
                            step.error = "Agent execution was aborted."
                            break
                        if not result.success:
                            step.status = "FAILED"
                            step.error = step.output or "Agent reported failure."
                            # Do not replay potentially side-effecting actions
                            # unless the verifier explicitly marks retry safe.
                            verification = self._verify(execution_task, result)
                            if attempt < self.max_attempts and verification.retryable:
                                continue
                            break

                        verification = self._verify(execution_task, result)
                        step.verification_reason = verification.reason
                        if verification.verified:
                            step.status = "VERIFIED"
                            outputs[task.id] = step.output
                            break
                        step.status = "UNVERIFIED"
                        step.error = verification.reason or "Postcondition not verified."
                        if attempt < self.max_attempts and verification.retryable:
                            continue
                        break
                    except Exception as exc:
                        logger.exception("Verified executor failed on task %s", task.id)
                        step.status = "FAILED"
                        step.error = f"{type(exc).__name__}: {exc}"
                        break

                step.ended_at = time.time()
                self._checkpoint(report)

        report.ended_at = time.time()
        if report.success:
            report.status = "VERIFIED"
        elif any(step.status == "ABORTED" for step in report.steps):
            report.status = "ABORTED"
        elif any(step.status == "VERIFIED" for step in report.steps):
            report.status = "PARTIAL"
        else:
            report.status = "FAILED"
        self._checkpoint(report)
        return report

    def _verify(self, task: Task, result: TaskResult) -> VerificationResult:
        try:
            value = self.verifier(task, result)
            if isinstance(value, VerificationResult):
                return value
            # Strictly reject ambiguous verifier return types. Truthy strings
            # must never be mistaken for proof of completion.
            raise TypeError("Verifier must return VerificationResult")
        except Exception as exc:
            logger.exception("Postcondition verification failed for %s", task.id)
            return VerificationResult(False, f"Verifier error: {type(exc).__name__}: {exc}")

    @staticmethod
    def _validate_graph(tasks: list[Task]) -> None:
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique")
        known = set(ids)
        for task in tasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task(s): {', '.join(sorted(unknown))}"
                )
            if task.id in task.depends_on:
                raise ValueError(f"Task '{task.id}' cannot depend on itself")

        # Kahn's algorithm catches multi-node cycles before any side effects.
        remaining = {task.id: set(task.depends_on) for task in tasks}
        resolved: set[str] = set()
        while remaining:
            ready = {task_id for task_id, deps in remaining.items() if deps <= resolved}
            if not ready:
                raise ValueError("Task dependency graph contains a cycle")
            resolved.update(ready)
            for task_id in ready:
                remaining.pop(task_id)

    def _checkpoint(self, report: ExecutionReport) -> None:
        if self.checkpoint_path is None:
            return
        path = self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        with self._checkpoint_lock:
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
