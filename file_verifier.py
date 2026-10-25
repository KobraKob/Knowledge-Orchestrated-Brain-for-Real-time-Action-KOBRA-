"""Independent postcondition checks for filesystem actions."""
from __future__ import annotations

import os
import re
from pathlib import Path

from models import Task, TaskResult
from verified_executor import VerificationResult

# DevAgent emits this exact prefix after its create_file tool writes a file.
_FILE_CREATED = re.compile(r"(?im)^File created:\s*(.+?)\s*$")


def verify_file_creation(task: Task, result: TaskResult) -> VerificationResult:
    """Confirm a reported file creation by inspecting the real filesystem.

    The agent's output is used only to locate the claimed path; it is never
    treated as proof. Verification succeeds only when the path resolves to a
    regular file that can be opened and read as UTF-8.
    """
    if not result.success:
        return VerificationResult(False, "Agent reported failure; file creation cannot be verified.")

    match = _FILE_CREATED.search(result.output or "")
    if not match:
        return VerificationResult(
            False,
            "Agent output did not identify a file created by the supported file tool.",
        )

    raw_path = match.group(1).strip().strip('"')
    if not raw_path:
        return VerificationResult(False, "No file path was provided for verification.")

    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    try:
        if not path.is_file():
            return VerificationResult(False, f"Expected file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            handle.read()
    except (OSError, UnicodeError, ValueError) as exc:
        return VerificationResult(False, f"File is not readable as UTF-8: {type(exc).__name__}: {exc}")

    return VerificationResult(True, f"Verified file exists and is readable: {path}")
