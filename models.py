"""
models.py — Shared data models for the KOBRA multi-agent system.

Task        : a single unit of work routed to one agent
TaskResult  : the outcome after an agent executes a task
"""

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str                                         # e.g. "t1", "t2"
    agent_name: str                                 # matches key in agent registry
    instruction: str                                # what this agent must do
    can_parallelize: bool = True                    # safe to run alongside other tasks?
    depends_on: list[str] = field(default_factory=list)  # task IDs that must finish first
    injected_context: str = ""                      # filled by task_queue from dependency results


@dataclass
class TaskResult:
    task_id: str
    agent_name: str
    success: bool
    output: str                                     # result string or error message
    duration_seconds: float = 0.0
    was_aborted: bool = False
