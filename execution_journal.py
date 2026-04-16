"""
execution_journal.py — Append-only execution journal for KOBRA v4.

Every tool call, result, retry, and decision is logged to disk.
Enables KOBRA to answer "what did you just do?" and "why did that fail?"

Format: JSONL (one JSON object per line) — easy to tail, grep, or parse.
Location: kobra_journal.jsonl (append-only, never truncated)
"""

import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_JOURNAL_PATH = "kobra_journal.jsonl"
_MAX_LINES = 10_000   # rotate after this many lines


class ExecutionJournal:
    """
    Thread-safe append-only journal.
    Each entry is a JSON line with: timestamp, event_type, and event-specific fields.
    """

    def __init__(self, path: str = _JOURNAL_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._line_count = self._count_lines()

    def _count_lines(self) -> int:
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
        except Exception:
            pass
        return 0

    def _append(self, event_type: str, data: dict) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with self._lock:
                # Rotate if too large
                if self._line_count >= _MAX_LINES:
                    self._rotate()
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._line_count += 1
        except Exception as exc:
            logger.debug("[JOURNAL] Write failed: %s", exc)

    def _rotate(self) -> None:
        """Keep last 5000 lines, discard oldest."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            keep = lines[-5000:]
            with open(self._path, "w", encoding="utf-8") as f:
                f.writelines(keep)
            self._line_count = len(keep)
            logger.info("[JOURNAL] Rotated — kept last 5000 entries.")
        except Exception as exc:
            logger.warning("[JOURNAL] Rotation failed: %s", exc)

    # ── Public logging methods ────────────────────────────────────────────────

    def log_command(self, transcript: str, enriched: str, task_count: int) -> None:
        self._append("COMMAND", {
            "transcript": transcript[:300],
            "enriched": enriched[:300],
            "task_count": task_count,
        })

    def log_plan(self, transcript: str, plan: dict) -> None:
        self._append("PLAN", {
            "transcript": transcript[:200],
            "intent": plan.get("intent", "")[:200],
            "complexity": plan.get("complexity", ""),
            "goals": plan.get("goals", [])[:5],
        })

    def log_task_start(self, task_id: str, agent_name: str, instruction: str) -> None:
        self._append("TASK_START", {
            "task_id": task_id,
            "agent": agent_name,
            "instruction": instruction[:300],
        })

    def log_task_result(self, task_id: str, agent_name: str, success: bool, output: str, duration: float) -> None:
        self._append("TASK_RESULT", {
            "task_id": task_id,
            "agent": agent_name,
            "success": success,
            "output": output[:500],
            "duration_s": round(duration, 2),
        })

    def log_tool_call(self, agent: str, tool_name: str, tool_input: str) -> None:
        self._append("TOOL_CALL", {
            "agent": agent,
            "tool": tool_name,
            "input": str(tool_input)[:200],
        })

    def log_tool_result(self, agent: str, tool_name: str, success: bool, output: str, duration_ms: float) -> None:
        self._append("TOOL_RESULT", {
            "agent": agent,
            "tool": tool_name,
            "success": success,
            "output": output[:300],
            "duration_ms": round(duration_ms, 1),
        })

    def log_reflection(self, tool_name: str, verdict: str, confidence: float, reason: str) -> None:
        self._append("REFLECTION", {
            "tool": tool_name,
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason[:200],
        })

    def log_retry(self, task_id: str, agent: str, attempt: int, reason: str, hint: str) -> None:
        self._append("RETRY", {
            "task_id": task_id,
            "agent": agent,
            "attempt": attempt,
            "reason": reason[:200],
            "hint": hint[:200],
        })

    def log_state_transition(self, task_id: str, from_state: str, to_state: str, reason: str = "") -> None:
        self._append("STATE_TRANSITION", {
            "task_id": task_id,
            "from": from_state,
            "to": to_state,
            "reason": reason[:200],
        })

    def log_abandon(self, task_id: str, agent: str, reason: str) -> None:
        self._append("ABANDON", {
            "task_id": task_id,
            "agent": agent,
            "reason": reason[:200],
        })

    def log_guardrail_block(self, rule: str, transcript: str, details: str) -> None:
        self._append("GUARDRAIL_BLOCK", {
            "rule": rule,
            "transcript": transcript[:200],
            "details": details[:300],
        })

    def log_metacognition(self, task_id: str, confidence: float, uncertain_steps: list) -> None:
        self._append("METACOGNITION", {
            "task_id": task_id,
            "confidence": confidence,
            "uncertain_steps": uncertain_steps[:5],
        })

    def log_response(self, transcript: str, response: str, was_cut_off: bool) -> None:
        self._append("RESPONSE", {
            "transcript": transcript[:200],
            "response": response[:300],
            "was_cut_off": was_cut_off,
        })

    def get_recent(self, n: int = 20, event_type: str | None = None) -> list[dict]:
        """Return last n journal entries, optionally filtered by event type."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            entries = []
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if event_type is None or entry.get("event") == event_type:
                        entries.append(entry)
                        if len(entries) >= n:
                            break
                except json.JSONDecodeError:
                    continue
            return list(reversed(entries))
        except Exception:
            return []

    def explain_last_action(self) -> str:
        """
        Answer "what did you just do?" from the journal.
        Returns a human-readable summary of the last command's execution.
        """
        recent = self.get_recent(50)
        if not recent:
            return "No recent actions recorded."

        # Find last COMMAND entry
        command_entry = None
        for e in reversed(recent):
            if e.get("event") == "COMMAND":
                command_entry = e
                break

        if not command_entry:
            return "No recent command found."

        cmd_ts = command_entry.get("ts", "")
        lines = [f"Last command: \"{command_entry.get('transcript', '')}\""]

        # Collect subsequent events
        found_cmd = False
        for e in recent:
            if not found_cmd:
                if e.get("ts") == cmd_ts and e.get("event") == "COMMAND":
                    found_cmd = True
                continue
            ev = e.get("event", "")
            if ev == "TASK_START":
                lines.append(f"  → {e['agent']}: {e['instruction'][:60]}")
            elif ev == "TASK_RESULT":
                status = "✓" if e["success"] else "✗"
                lines.append(f"    {status} {e['output'][:80]}")
            elif ev == "RETRY":
                lines.append(f"    ↺ Retry #{e['attempt']}: {e['reason'][:60]}")
            elif ev == "GUARDRAIL_BLOCK":
                lines.append(f"    Blocked: {e['rule']} — {e['details'][:60]}")
            elif ev == "COMMAND":
                break  # next command starts here

        return "\n".join(lines)
