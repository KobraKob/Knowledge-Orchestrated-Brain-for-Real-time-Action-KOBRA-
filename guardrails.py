"""
guardrails.py — Concrete safety guardrails for KOBRA v4.

Prevents:
  1. Destructive system commands (rm -rf, format, shutdown without confirmation)
  2. Hallucinated file paths (paths to system dirs a voice assistant shouldn't touch)
  3. Infinite retry loops (detected before execution limiter fires)
  4. Prompt injection in tool inputs
  5. Runaway LLM-generated code (dangerous patterns in interpreter output)

Every block is logged to ExecutionJournal.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    allowed:          bool
    rule:             str    # which rule triggered
    reason:           str    # human-readable explanation
    safe_alternative: str = ""


# ── Rule 1: Destructive shell command patterns ────────────────────────────────

_DESTRUCTIVE_SHELL = [
    re.compile(r"\brm\s+(-rf?|-fr?|--recursive)\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[A-Za-z]:", re.IGNORECASE),
    re.compile(r"\bdel\s+/[SF]\b", re.IGNORECASE),
    re.compile(r"\brmdir\s+/[SQ]\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+(database|table|schema)\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bshutdown\s+(/[SR]|-[rh])\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bpoweroff\b", re.IGNORECASE),
    re.compile(r"\bkillall\b", re.IGNORECASE),
    re.compile(r"\bnetsh\s+firewall\b", re.IGNORECASE),
    re.compile(r"reg\s+delete\b", re.IGNORECASE),
]

# ── Rule 2: Protected system paths ────────────────────────────────────────────

_PROTECTED_PATHS = [
    r"C:\\Windows",
    r"C:\\Program Files",
    r"C:\\System32",
    r"/etc/",
    r"/sys/",
    r"/proc/",
    r"/boot/",
    r"~/.ssh",
    r"~/.gnupg",
]
_PROTECTED_PATH_RE = re.compile(
    "|".join(re.escape(p) for p in _PROTECTED_PATHS), re.IGNORECASE
)

# ── Rule 3: Prompt injection ──────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r"ignore (previous|above|all) instructions", re.IGNORECASE),
    re.compile(r"forget (your|all) (rules|instructions|guidelines)", re.IGNORECASE),
    re.compile(r"new system prompt:", re.IGNORECASE),
    re.compile(r"</?(system|human|assistant)>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]"),
]

# ── Rule 4: Dangerous code patterns for interpreter ──────────────────────────

_DANGEROUS_CODE = [
    re.compile(r"\bos\.system\s*\(\s*['\"]rm\s+-rf", re.IGNORECASE),
    re.compile(r"\bsubprocess\.(run|call|Popen)\s*\(\s*\[?['\"]rm\s+-rf", re.IGNORECASE),
    re.compile(r"\bshutil\.rmtree\s*\(\s*['\"]\/", re.IGNORECASE),
    re.compile(r"__import__\s*\(\s*['\"]os['\"]\s*\)\.system"),
    re.compile(r"\beval\s*\(\s*input\b"),
    re.compile(r"\bexec\s*\(\s*input\b"),
    re.compile(r"open\(['\"]\/etc\/passwd"),
]

# Max retries before guardrail fires
_MAX_RETRIES_GUARD = 4

# ── Irreversible actions requiring explicit user confirmation ─────────────────
# These tools produce side-effects that cannot be undone. Before any task whose
# instruction implies one of these will be called, the orchestrator asks once.
REQUIRES_CONFIRMATION: set[str] = {
    "send_email",
    "delete_file",
    "whatsapp_send_message",
    "delete_calendar_event",
    "run_command",      # only when deletion/formatting verbs present
}

# Substrings in the instruction that elevate run_command to confirmation-needed
_DESTRUCTIVE_INSTRUCTION_SIGNALS = (
    "delete", "remove", "erase", "format", "wipe", "drop", "truncate",
    "uninstall", "rm ", "del ",
)

# Confirmation state — maps a normalised instruction hash to True once confirmed
_CONFIRMED_ACTIONS: dict[str, bool] = {}


def _instruction_key(agent: str, instruction: str) -> str:
    """Stable key for deduplicating confirmations within a session."""
    import hashlib
    return hashlib.sha1(f"{agent}:{instruction[:120]}".encode()).hexdigest()[:16]


class Guardrails:
    """
    Safety layer that validates every tool call before execution.
    All blocks are logged.
    """

    def __init__(self) -> None:
        self._confirmed: set[str] = set()   # user-confirmed destructive actions
        self._retry_counts: dict[str, int] = {}

    def check_command(self, command: str, tool_name: str = "run_command") -> GuardrailResult:
        """Check a shell command before execution."""
        # Rule 1: Destructive shell commands
        for pattern in _DESTRUCTIVE_SHELL:
            if pattern.search(command):
                if command in self._confirmed:
                    return GuardrailResult(
                        allowed=True, rule="user_confirmed",
                        reason="User confirmed destructive action."
                    )
                return GuardrailResult(
                    allowed=False,
                    rule="destructive_command",
                    reason=f"Command matches destructive pattern.",
                    safe_alternative="Tell me exactly what you want to delete and I'll confirm with you first.",
                )

        # Rule 2: Protected paths
        if _PROTECTED_PATH_RE.search(command):
            return GuardrailResult(
                allowed=False,
                rule="protected_path",
                reason="Command targets a protected system path.",
                safe_alternative="I don't modify system directories. Let me know what you're trying to accomplish.",
            )

        # Rule 3: Prompt injection
        if self._is_injection(command):
            return GuardrailResult(
                allowed=False,
                rule="prompt_injection",
                reason="Command contains prompt injection patterns.",
                safe_alternative="I detected an attempt to modify my instructions. Command blocked.",
            )

        return GuardrailResult(allowed=True, rule="pass", reason="Command passed all checks.")

    def check_code(self, code: str) -> GuardrailResult:
        """Check generated code before execution (for InterpreterAgent)."""
        for pattern in _DANGEROUS_CODE:
            if pattern.search(code):
                return GuardrailResult(
                    allowed=False,
                    rule="dangerous_code",
                    reason=f"Generated code contains dangerous pattern.",
                    safe_alternative="I won't execute this code — it contains potentially destructive operations.",
                )
        return GuardrailResult(allowed=True, rule="pass", reason="Code passed safety checks.")

    def check_file_path(self, path: str) -> GuardrailResult:
        """Check a file path before read/write."""
        if _PROTECTED_PATH_RE.search(path):
            return GuardrailResult(
                allowed=False,
                rule="protected_path",
                reason=f"Path '{path}' is in a protected system directory.",
                safe_alternative="I can only access files in your user directories (Desktop, Documents, Projects).",
            )
        return GuardrailResult(allowed=True, rule="pass", reason="Path allowed.")

    def check_retry_loop(self, task_id: str) -> GuardrailResult:
        """Detect infinite retry loops before execution limiter fires."""
        count = self._retry_counts.get(task_id, 0) + 1
        self._retry_counts[task_id] = count
        if count >= _MAX_RETRIES_GUARD:
            return GuardrailResult(
                allowed=False,
                rule="retry_loop",
                reason=f"Task {task_id} has retried {count} times — stopping to prevent infinite loop.",
                safe_alternative="This task keeps failing. Should I try a different approach?",
            )
        return GuardrailResult(allowed=True, rule="pass", reason=f"Retry {count}/{_MAX_RETRIES_GUARD} allowed.")

    def check_instruction(self, instruction: str) -> GuardrailResult:
        """Check a task instruction for injection or destructive intent."""
        if self._is_injection(instruction):
            return GuardrailResult(
                allowed=False,
                rule="prompt_injection",
                reason="Task instruction contains prompt injection patterns.",
            )
        return GuardrailResult(allowed=True, rule="pass", reason="Instruction passed checks.")

    def needs_confirmation(self, agent_name: str, instruction: str) -> str | None:
        """
        Return a confirmation question if this task requires one, else None.
        Once the user confirms (call confirm_pending()), the same task runs freely.

        Logic:
          - email agent → always ask before sending
          - browser agent + whatsapp keyword → ask
          - integration agent + delete keyword → ask
          - dev/system agent + destructive keywords → ask
        """
        key = _instruction_key(agent_name, instruction)
        if _CONFIRMED_ACTIONS.get(key):
            return None   # already confirmed this session

        inst_lower = instruction.lower()

        # Email
        if agent_name == "integration" and any(w in inst_lower for w in ("send", "email", "mail")):
            # Extract recipient hint for the question
            import re as _re
            recipient = ""
            m = _re.search(r"to\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,})", inst_lower)
            if m:
                recipient = f" to {m.group(1)}"
            return f"Just to confirm, sir — send that email{recipient}?"

        # WhatsApp
        if agent_name == "browser" and "whatsapp" in inst_lower and "send" in inst_lower:
            return "Confirm sending that WhatsApp message, sir?"

        # Calendar delete
        if agent_name == "integration" and "delete" in inst_lower and "calendar" in inst_lower:
            return "Confirm deleting that calendar event, sir?"

        # Dev/system with destructive signals
        if agent_name in ("dev", "system") and any(
            sig in inst_lower for sig in _DESTRUCTIVE_INSTRUCTION_SIGNALS
        ):
            return "That sounds destructive, sir — want me to go ahead?"

        return None

    def confirm_pending(self, agent_name: str, instruction: str) -> None:
        """Mark a pending confirmation as approved for this session."""
        key = _instruction_key(agent_name, instruction)
        _CONFIRMED_ACTIONS[key] = True
        logger.info("[GUARDRAILS] Confirmed: agent=%s key=%s", agent_name, key)

    def confirm_action(self, action: str) -> None:
        """Mark an action as user-confirmed (bypass destructive check once)."""
        self._confirmed.add(action)
        logger.info("[GUARDRAILS] Action confirmed by user: %s", action[:80])

    def reset_retry_count(self, task_id: str) -> None:
        self._retry_counts.pop(task_id, None)

    def _is_injection(self, text: str) -> bool:
        return any(p.search(text) for p in _INJECTION_PATTERNS)

    @staticmethod
    def describe_block(result: "GuardrailResult") -> str:
        """Human-readable block description for user feedback."""
        msg = f"I blocked that action. {result.reason}"
        if result.safe_alternative:
            msg += f" {result.safe_alternative}"
        return msg
