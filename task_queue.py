"""
task_queue.py — Parallel and sequential task executor for KOBRA v4.

Algorithm:
  1. Build a dependency graph from the task list.
  2. Find the first wave: tasks with no outstanding dependencies.
  3. Submit parallel-safe tasks to ThreadPoolExecutor; sequential tasks run one by one.
  4. As tasks complete, inject their results into dependent tasks and unblock the next wave.
  5. Repeat until all tasks done or abort_flag is set.

v4 additions:
  - Reflection-driven retry (Reflector evaluates every task output)
  - Explicit state machine (StateTracker tracks PENDING/RUNNING/SUCCESS/FAILED/RETRYING/ABANDONED/SKIPPED)
  - Append-only execution journal (ExecutionJournal logs every event to disk)
  - Dependency-aware skip propagation (if dep ABANDONED → dependent SKIPPED)
  - Partial-success warning surfaced after execution
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import concurrent.futures

import config
from models import Task, TaskResult
from reflection import Reflector, ReflectionVerdict
from state_tracker import StateTracker, TaskState, MAX_RETRIES
from execution_journal import ExecutionJournal

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────

_reflector    = Reflector()
_state_tracker = StateTracker()
_journal      = ExecutionJournal()


def get_journal() -> ExecutionJournal:
    return _journal


def get_state_tracker() -> StateTracker:
    return _state_tracker


# ── TaskQueue ─────────────────────────────────────────────────────────────────

class TaskQueue:
    def execute(
        self,
        tasks: list[Task],
        agents: dict,
        abort_flag: threading.Event,
    ) -> list[TaskResult]:
        """
        Execute tasks respecting dependency order and parallelism flags.
        Returns all TaskResult objects in completion order.

        Public interface unchanged from v3. Internally delegates to
        _execute_with_state() which adds reflection, state tracking, and
        journal logging.
        """
        if not tasks:
            return []

        # Fast path: single task, no deps
        if len(tasks) == 1:
            task = tasks[0]
            agent = agents.get(task.agent_name)
            if agent is None:
                return [TaskResult(
                    task_id=task.id, agent_name=task.agent_name,
                    success=False, output=f"No agent for '{task.agent_name}'.",
                )]
            state = _state_tracker.new_execution(task.instruction)
            state.add_step(task.id, task.agent_name, task.instruction)
            _journal.log_command(task.instruction, task.instruction, 1)
            result = self._execute_with_retry(task, agent, abort_flag, state, _journal)
            warning = _state_tracker.get_partial_state_warning()
            if warning:
                logger.warning("[QUEUE] %s", warning)
            return [result]

        # Multi-task path
        state = _state_tracker.new_execution(tasks[0].instruction if tasks else "")
        _journal.log_command(
            tasks[0].instruction if tasks else "",
            tasks[0].instruction if tasks else "",
            len(tasks),
        )
        results = self._execute_with_state(tasks, agents, abort_flag, state, _journal)

        # Surface partial-success warning
        warning = _state_tracker.get_partial_state_warning()
        if warning:
            logger.warning("[QUEUE] %s", warning)

        return results

    # ── Core execution with state + reflection ────────────────────────────────

    def _execute_with_state(
        self,
        tasks: list[Task],
        agents: dict,
        abort_flag: threading.Event,
        state,
        journal: ExecutionJournal,
    ) -> list[TaskResult]:
        """
        Execute tasks respecting dependency graph.
        Parallel where safe (task.can_parallelize=True), sequential where not.
        Skips tasks whose dependencies were abandoned.
        """
        # Register all tasks as PENDING
        for task in tasks:
            state.add_step(task.id, task.agent_name, task.instruction)

        results: list[TaskResult] = []
        completed_ids: set[str] = set()
        result_map: dict[str, str] = {}     # task_id → output (for context injection)
        pending: dict[str, Task] = {t.id: t for t in tasks}

        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_AGENTS) as executor:
            while pending and not abort_flag.is_set():
                # Collect IDs of abandoned steps (failed deps)
                abandoned_ids = {
                    tid for tid, rec in state.steps.items()
                    if rec.state == TaskState.ABANDONED
                }

                # Skip tasks whose dependencies were abandoned
                to_skip = [
                    t for t in pending.values()
                    if any(dep in abandoned_ids for dep in t.depends_on)
                ]
                for task in to_skip:
                    state.transition(task.id, TaskState.SKIPPED, error="Dependency abandoned.")
                    journal.log_state_transition(task.id, "PENDING", "SKIPPED", "Dependency abandoned.")
                    results.append(TaskResult(
                        task_id=task.id, agent_name=task.agent_name,
                        success=False, output="Skipped — dependency failed.",
                    ))
                    completed_ids.add(task.id)
                    del pending[task.id]

                # Find tasks whose deps are fully satisfied
                ready = [
                    t for t in pending.values()
                    if all(dep in completed_ids for dep in t.depends_on)
                ]

                if not ready:
                    if pending:
                        logger.error("[QUEUE] Deadlock detected — breaking.")
                    break

                # Inject dependency outputs into ready tasks
                for task in ready:
                    self._inject_context(task, result_map)

                # Split into parallel vs sequential
                parallel   = [t for t in ready if t.can_parallelize]
                sequential = [t for t in ready if not t.can_parallelize]

                # ── Parallel tasks ────────────────────────────────────────────
                futures: dict[Future, Task] = {}
                for task in parallel:
                    del pending[task.id]
                    state.transition(task.id, TaskState.RUNNING)
                    journal.log_task_start(task.id, task.agent_name, task.instruction)

                    agent = agents.get(task.agent_name)
                    if agent is None:
                        state.transition(task.id, TaskState.FAILED,
                                         error=f"Agent '{task.agent_name}' not found.")
                        journal.log_abandon(task.id, task.agent_name, "Agent not registered.")
                        tr = TaskResult(
                            task_id=task.id, agent_name=task.agent_name,
                            success=False, output=f"No agent registered for '{task.agent_name}'.",
                        )
                        results.append(tr)
                        completed_ids.add(task.id)
                        result_map[task.id] = tr.output
                        continue

                    fut = executor.submit(
                        self._execute_with_retry, task, agent, abort_flag, state, journal
                    )
                    futures[fut] = task

                # Collect parallel results
                if futures:
                    try:
                        timeout = config.AGENT_TIMEOUT * len(futures)
                        for fut in as_completed(futures, timeout=timeout):
                            task = futures[fut]
                            try:
                                result: TaskResult = fut.result(timeout=1)
                            except Exception as exc:
                                logger.error("[QUEUE] Future for task %s raised: %s", task.id, exc)
                                state.transition(task.id, TaskState.FAILED, error=str(exc))
                                result = TaskResult(
                                    task_id=task.id, agent_name=task.agent_name,
                                    success=False, output=f"Agent raised: {exc}",
                                )
                            results.append(result)
                            completed_ids.add(task.id)
                            result_map[task.id] = result.output or ""
                            if abort_flag.is_set():
                                break
                    except concurrent.futures.TimeoutError:
                        logger.warning("[QUEUE] Timeout waiting for agents — collecting partial results.")
                        for fut, task in futures.items():
                            if task.id in completed_ids:
                                continue
                            if fut.done():
                                try:
                                    result = fut.result()
                                    results.append(result)
                                    completed_ids.add(task.id)
                                    result_map[task.id] = result.output or ""
                                except Exception as exc:
                                    state.transition(task.id, TaskState.ABANDONED, error="Timed out.")
                                    journal.log_abandon(task.id, task.agent_name, "Timed out.")
                                    tr = TaskResult(
                                        task_id=task.id, agent_name=task.agent_name,
                                        success=False, output=f"Agent error: {exc}",
                                    )
                                    results.append(tr)
                                    completed_ids.add(task.id)
                                    result_map[task.id] = tr.output
                            else:
                                fut.cancel()
                                state.transition(task.id, TaskState.ABANDONED, error="Timed out.")
                                journal.log_abandon(task.id, task.agent_name, "Timed out.")
                                tr = TaskResult(
                                    task_id=task.id, agent_name=task.agent_name,
                                    success=False, output="Agent timed out.",
                                )
                                results.append(tr)
                                completed_ids.add(task.id)
                                result_map[task.id] = tr.output

                # ── Sequential tasks ──────────────────────────────────────────
                for task in sequential:
                    if abort_flag.is_set():
                        break
                    del pending[task.id]
                    state.transition(task.id, TaskState.RUNNING)
                    journal.log_task_start(task.id, task.agent_name, task.instruction)

                    agent = agents.get(task.agent_name)
                    if agent is None:
                        state.transition(task.id, TaskState.FAILED,
                                         error=f"Agent '{task.agent_name}' not found.")
                        journal.log_abandon(task.id, task.agent_name, "Agent not registered.")
                        tr = TaskResult(
                            task_id=task.id, agent_name=task.agent_name,
                            success=False, output=f"No agent registered for '{task.agent_name}'.",
                        )
                        results.append(tr)
                        completed_ids.add(task.id)
                        result_map[task.id] = tr.output
                        continue

                    result = self._execute_with_retry(task, agent, abort_flag, state, journal)
                    results.append(result)
                    completed_ids.add(task.id)
                    result_map[task.id] = result.output or ""

        # Mark any tasks still pending as aborted
        for task_id, task in pending.items():
            state.transition(task_id, TaskState.SKIPPED, error="Aborted before execution.")
            results.append(TaskResult(
                task_id=task_id,
                agent_name=task.agent_name,
                success=False,
                output="Aborted before execution.",
                was_aborted=True,
            ))

        return results

    def _execute_with_retry(
        self,
        task: Task,
        agent,
        abort_flag: threading.Event,
        state,
        journal: ExecutionJournal,
    ) -> TaskResult:
        """
        Execute a single task with reflection-driven retry logic.
        Handles PENDING→RUNNING→SUCCESS/FAILED→RETRYING→RUNNING→... transitions.

        Note: caller is responsible for the initial RUNNING transition when
        going through _execute_with_state. For the single-task fast path,
        this method handles it internally.
        """
        for attempt in range(1, MAX_RETRIES + 2):  # +2: initial attempt + up to MAX_RETRIES retries
            if abort_flag.is_set():
                state.transition(task.id, TaskState.ABANDONED, error="Aborted.")
                journal.log_abandon(task.id, task.agent_name, "Aborted by abort_flag.")
                return TaskResult(
                    task_id=task.id, agent_name=task.agent_name,
                    success=False, output="Aborted.",
                )

            # For retry attempts, we need to re-enter RUNNING from RETRYING
            if attempt > 1:
                state.transition(task.id, TaskState.RUNNING)
                journal.log_task_start(task.id, task.agent_name, task.instruction)

            start = time.time()
            try:
                result: TaskResult = agent.execute(task, abort_flag)
            except Exception as exc:
                logger.error("[QUEUE] Agent %s raised on attempt %d: %s", task.agent_name, attempt, exc)
                result = TaskResult(
                    task_id=task.id, agent_name=task.agent_name,
                    success=False, output=f"Agent exception: {exc}",
                )
            duration = time.time() - start

            journal.log_task_result(
                task.id, task.agent_name, result.success, result.output or "", duration
            )

            # Reflect on the result
            reflection = _reflector.reflect(
                tool_name=task.agent_name,
                tool_output=result,
            )
            journal.log_reflection(
                task.agent_name, reflection.verdict.value, reflection.confidence, reflection.reason
            )

            # SUCCESS path
            if reflection.verdict == ReflectionVerdict.SUCCESS or result.success:
                state.transition(task.id, TaskState.SUCCESS, output=result.output or "")
                return result

            # FAILURE path — check if we should retry
            if reflection.should_retry and state.can_retry(task.id):
                state.transition(
                    task.id, TaskState.FAILED,
                    error=reflection.reason,
                    retry_hint=reflection.retry_hint,
                )
                journal.log_retry(
                    task.id, task.agent_name, attempt,
                    reflection.reason, reflection.retry_hint,
                )
                state.transition(task.id, TaskState.RETRYING)

                # Append retry hint to instruction for next attempt
                if reflection.retry_hint:
                    task.instruction = f"{task.instruction}\n[Retry hint: {reflection.retry_hint}]"
                continue  # loop back for retry

            # No retry — terminal failure
            final_state = TaskState.ABANDONED if attempt > 1 else TaskState.FAILED
            state.transition(task.id, final_state, error=reflection.reason)
            journal.log_abandon(task.id, task.agent_name, reflection.reason)
            return result

        # Retries exhausted
        state.transition(task.id, TaskState.ABANDONED, error="Retries exhausted.")
        journal.log_abandon(task.id, task.agent_name, "Retries exhausted.")
        return TaskResult(
            task_id=task.id, agent_name=task.agent_name,
            success=False,
            output=f"Failed after {MAX_RETRIES} retries.",
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_ready(pending: dict[str, Task], completed: set[str]) -> list[Task]:
        """Return tasks from pending whose dependencies are all in completed."""
        return [
            task for task in pending.values()
            if all(dep in completed for dep in task.depends_on)
        ]

    @staticmethod
    def _inject_context(task: Task, result_map: dict[str, str]) -> None:
        """Append dependency outputs to task.injected_context before execution."""
        if not task.depends_on:
            return
        snippets: list[str] = []
        for dep_id in task.depends_on:
            if dep_id in result_map:
                snippets.append(f"[Result of {dep_id}]: {result_map[dep_id]}")
        if snippets:
            task.injected_context = "\n".join(snippets)
