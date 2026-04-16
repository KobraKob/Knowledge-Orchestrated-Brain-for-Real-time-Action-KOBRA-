"""
reflection.py — Concrete reflection rubrics for KOBRA v4.

Every tool type has explicit, rule-based success criteria.
LLM is used ONLY as a last resort for ambiguous natural-language outputs.

Rubric priority:
  1. Exit code / HTTP status (binary, always authoritative)
  2. Known error strings (fast string matching)
  3. Empty/null result detection
  4. LLM confidence only for ambiguous text outputs (fallback, not default)
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ReflectionVerdict(Enum):
    SUCCESS   = "success"
    FAILURE   = "failure"
    UNCERTAIN = "uncertain"   # triggers LLM fallback
    RETRY     = "retry"       # transient failure, worth retrying


@dataclass
class ReflectionResult:
    verdict:    ReflectionVerdict
    confidence: float          # 0.0 – 1.0, rule-based
    reason:     str
    should_retry: bool = False
    retry_hint:   str  = ""    # hint for next attempt


# ── Known error string patterns (fast, rule-based) ────────────────────────────

_ERROR_PATTERNS = [
    r"error:",
    r"exception:",
    r"traceback \(most recent call last\)",
    r"no such file or directory",
    r"permission denied",
    r"command not found",
    r"timed out",
    r"connection refused",
    r"[Ff]ailed to",
    r"unable to",
    r"not found",
    r"does not exist",
    r"invalid",
    r"unauthorized",
    r"403 forbidden",
    r"404 not found",
    r"500 internal server error",
]

_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)

_SUCCESS_INDICATORS = [
    "successfully", "created", "opened", "sent", "saved", "completed",
    "done", "ok", "found", "retrieved", "playing", "launched",
]

# Transient errors worth retrying
_RETRY_PATTERNS = [
    r"timed out", r"connection refused", r"network", r"temporarily",
    r"rate limit", r"429", r"503", r"try again",
]
_RETRY_RE = re.compile("|".join(_RETRY_PATTERNS), re.IGNORECASE)


class Reflector:
    """
    Evaluates tool outputs using tool-specific rubrics.
    Returns a ReflectionResult with verdict + confidence + retry guidance.
    """

    def reflect(
        self,
        tool_name: str,
        tool_output: Any,
        exit_code: int | None = None,
        stderr: str | None = None,
        http_status: int | None = None,
        extra: dict | None = None,
    ) -> ReflectionResult:
        """
        Main entry point. Dispatches to tool-specific rubric.
        Falls back to generic text analysis if no specific rubric exists.
        """
        extra = extra or {}

        # Normalize output to string for pattern matching
        if hasattr(tool_output, "output"):
            output_str = str(tool_output.output or "")
            if hasattr(tool_output, "success") and tool_output.success is False:
                # ToolResult explicitly marked as failure
                return ReflectionResult(
                    verdict=ReflectionVerdict.FAILURE,
                    confidence=1.0,
                    reason="ToolResult.success=False",
                    should_retry=_RETRY_RE.search(output_str) is not None,
                    retry_hint="Try an alternative approach." if not _RETRY_RE.search(output_str) else "Retry — transient failure.",
                )
        else:
            output_str = str(tool_output or "")

        # Dispatch to tool-specific rubric
        rubric_map = {
            "run_command":           self._reflect_run_command,
            "run_shell":             self._reflect_run_command,
            "execute_command":       self._reflect_run_command,
            "web_search":            self._reflect_web_search,
            "search_web":            self._reflect_web_search,
            "click_element":         self._reflect_click,
            "click":                 self._reflect_click,
            "send_email":            self._reflect_send_email,
            "create_calendar_event": self._reflect_calendar,
            "get_calendar_events":   self._reflect_calendar,
            "play_spotify":          self._reflect_spotify,
            "control_spotify":       self._reflect_spotify,
            "create_file":           self._reflect_file_op,
            "read_file":             self._reflect_file_op,
            "take_screenshot":       self._reflect_screenshot,
            "scrape_page":           self._reflect_scrape,
            "navigate":              self._reflect_navigate,
            "send_whatsapp_message": self._reflect_whatsapp,
        }

        rubric_fn = rubric_map.get(tool_name)
        if rubric_fn:
            return rubric_fn(output_str, exit_code, stderr, http_status, extra)
        else:
            return self._reflect_generic(tool_name, output_str, exit_code, stderr)

    # ── Tool-specific rubrics ─────────────────────────────────────────────────

    def _reflect_run_command(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        """Shell command: exit code 0 + empty stderr = success. Authoritative."""
        if exit_code is not None:
            if exit_code == 0:
                # Check stderr for warnings that indicate real problems
                if stderr and len(stderr.strip()) > 20 and _ERROR_RE.search(stderr):
                    return ReflectionResult(
                        verdict=ReflectionVerdict.FAILURE,
                        confidence=0.9,
                        reason=f"Exit 0 but stderr contains errors: {stderr[:100]}",
                        should_retry=False,
                    )
                return ReflectionResult(
                    verdict=ReflectionVerdict.SUCCESS,
                    confidence=1.0,
                    reason=f"Exit code 0. Stderr: {len(stderr or '')} bytes.",
                )
            else:
                is_transient = _RETRY_RE.search(output or "") is not None or _RETRY_RE.search(stderr or "") is not None
                return ReflectionResult(
                    verdict=ReflectionVerdict.FAILURE,
                    confidence=1.0,
                    reason=f"Exit code {exit_code}. Stderr: {(stderr or '')[:150]}",
                    should_retry=is_transient,
                    retry_hint="Retry with corrected command." if not is_transient else "Retry — transient failure.",
                )

        # No exit code — fall back to output analysis
        return self._reflect_generic("run_command", output, exit_code, stderr)

    def _reflect_web_search(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        """Web search: has results = success. Error string or empty = failure."""
        if not output or len(output.strip()) < 20:
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE,
                confidence=0.95,
                reason="Empty or near-empty search results.",
                should_retry=True,
                retry_hint="Rephrase the search query.",
            )
        if _ERROR_RE.search(output):
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE,
                confidence=0.85,
                reason=f"Search returned error string: {output[:100]}",
                should_retry=_RETRY_RE.search(output) is not None,
                retry_hint="Try a different search query.",
            )
        # Has content
        return ReflectionResult(
            verdict=ReflectionVerdict.SUCCESS,
            confidence=0.95,
            reason=f"Search returned {len(output)} chars of results.",
        )

    def _reflect_click(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        """Click: success string or screenshot change indicator."""
        if "clicked" in output.lower() or "success" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.9, reason="Click confirmed.")
        if _ERROR_RE.search(output):
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=0.85,
                reason=f"Click error: {output[:100]}",
                should_retry=True, retry_hint="Re-identify the target element.",
            )
        # Ambiguous — use generic
        return self._reflect_generic("click", output, exit_code, stderr)

    def _reflect_send_email(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        """Email: HTTP 200 from Gmail API or 'sent' confirmation."""
        if http_status is not None:
            if http_status == 200:
                return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=1.0, reason="Gmail API returned 200.")
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=1.0,
                reason=f"Gmail API returned HTTP {http_status}.",
                should_retry=http_status in (429, 503, 500),
            )
        if "sent" in output.lower() or "delivered" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.95, reason="Email sent confirmation.")
        if _ERROR_RE.search(output):
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=0.9,
                reason=f"Email error: {output[:100]}",
                should_retry=_RETRY_RE.search(output) is not None,
            )
        return self._reflect_generic("send_email", output, exit_code, stderr)

    def _reflect_calendar(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if http_status == 200:
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=1.0, reason="Calendar API 200.")
        if "event" in output.lower() and ("created" in output.lower() or "found" in output.lower() or "scheduled" in output.lower()):
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.9, reason="Calendar event confirmed.")
        if not output or "no events" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.85, reason="No events — valid empty result.")
        return self._reflect_generic("calendar", output, exit_code, stderr)

    def _reflect_spotify(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if "playing" in output.lower() or "paused" in output.lower() or "now playing" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.95, reason="Spotify playback confirmed.")
        if "no active device" in output.lower():
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=1.0,
                reason="No active Spotify device.",
                should_retry=False,
                retry_hint="Open Spotify on a device first.",
            )
        return self._reflect_generic("spotify", output, exit_code, stderr)

    def _reflect_file_op(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if "created" in output.lower() or "saved" in output.lower() or "written" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.95, reason="File operation confirmed.")
        if _ERROR_RE.search(output):
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=0.9,
                reason=f"File op error: {output[:100]}",
                should_retry=False,
            )
        return self._reflect_generic("file_op", output, exit_code, stderr)

    def _reflect_screenshot(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if "screenshot" in output.lower() and ("saved" in output.lower() or "taken" in output.lower()):
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.95, reason="Screenshot confirmed.")
        return self._reflect_generic("screenshot", output, exit_code, stderr)

    def _reflect_scrape(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if not output or len(output.strip()) < 50:
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=0.9,
                reason="Scrape returned too little content.",
                should_retry=True, retry_hint="Try a different URL or selector.",
            )
        if _ERROR_RE.search(output[:200]):
            return ReflectionResult(verdict=ReflectionVerdict.FAILURE, confidence=0.85, reason=f"Scrape error: {output[:100]}")
        return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.9, reason=f"Scraped {len(output)} chars.")

    def _reflect_navigate(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if http_status and http_status >= 400:
            return ReflectionResult(verdict=ReflectionVerdict.FAILURE, confidence=1.0, reason=f"HTTP {http_status}.")
        if "navigated" in output.lower() or "loaded" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.9, reason="Navigation confirmed.")
        return self._reflect_generic("navigate", output, exit_code, stderr)

    def _reflect_whatsapp(self, output, exit_code, stderr, http_status, extra) -> ReflectionResult:
        if "sent" in output.lower() or "delivered" in output.lower() or "message sent" in output.lower():
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.95, reason="WhatsApp send confirmed.")
        if _ERROR_RE.search(output):
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE, confidence=0.85,
                reason=f"WhatsApp error: {output[:100]}",
                should_retry=True, retry_hint="Re-check contact name or WhatsApp session.",
            )
        return self._reflect_generic("whatsapp", output, exit_code, stderr)

    def _reflect_generic(self, tool_name, output, exit_code, stderr) -> ReflectionResult:
        """Fallback: pattern-match known error/success strings."""
        if not output:
            return ReflectionResult(
                verdict=ReflectionVerdict.UNCERTAIN, confidence=0.5,
                reason="Empty output — outcome unknown.",
                should_retry=True,
            )
        lower = output.lower()
        if _ERROR_RE.search(lower):
            is_transient = _RETRY_RE.search(lower) is not None
            return ReflectionResult(
                verdict=ReflectionVerdict.FAILURE,
                confidence=0.75,
                reason=f"Error pattern in output: {output[:100]}",
                should_retry=is_transient,
                retry_hint="Retry with different parameters." if is_transient else "Try a different approach.",
            )
        if any(s in lower for s in _SUCCESS_INDICATORS):
            return ReflectionResult(verdict=ReflectionVerdict.SUCCESS, confidence=0.8, reason="Success indicator found.")
        # Truly ambiguous
        return ReflectionResult(
            verdict=ReflectionVerdict.UNCERTAIN,
            confidence=0.4,
            reason="Output ambiguous — no clear success or error signal.",
        )
