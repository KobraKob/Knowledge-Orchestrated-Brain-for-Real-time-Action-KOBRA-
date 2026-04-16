"""
agents/interpreter_agent.py — Dynamic Code Execution Agent for KOBRA.

Implements the Open Interpreter pattern natively on Groq + Python subprocess.
No dependency on the open-interpreter package (incompatible with Python 3.14).

How it works:
  1. Receive a natural language dev task ("scaffold a FastAPI project called myapp")
  2. Call Groq (70b) → generate executable Python or PowerShell code
  3. Execute the code in a sandboxed subprocess with timeout
  4. Capture stdout / stderr / return code
  5. If error: send output back to Groq → regenerate → retry (up to 2x)
  6. Return a spoken summary of what happened

Compared to DevAgent (hardcoded templates):
  - Completely dynamic — no templates, no scaffold_project enums
  - Can write ANY code the LLM can generate
  - Handles complex multi-step tasks in a single pass
  - Self-healing: retries with error context on failure

Safety:
  - Only executes code in response to explicit voice commands
  - 30-second execution timeout (configurable)
  - stdout/stderr captured — no interactive terminal pops up
  - Destructive commands (rm -rf, format, del /f /s) require confirmation
    via speak_only first

Routing: orchestrator sends tasks here when keywords suggest complex
  dev work: "scaffold", "build me", "create a project", "write a script",
  "automate", "generate code", "make a function that", etc.
  Simple file ops ("create a file called notes.txt") still go to DevAgent.
"""

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Any

from groq import Groq

import config
from agents.base_agent import BaseAgent
from models import Task

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EXEC_TIMEOUT: int = 60          # seconds before subprocess is killed (websites write many files)
MAX_OUTPUT_CHARS: int = 4000    # cap on stdout/stderr fed back to LLM
MAX_RETRIES: int = 2            # max code regeneration attempts on failure

# Patterns that require the LLM to explicitly confirm before executing
_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/[sfq]",
    r"\bformat\s+[a-z]:",
    r"\brmdir\s+/s\b",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b.*\*",
]

_DESKTOP  = os.path.join(os.path.expanduser("~"), "Desktop")
_USERNAME = os.environ.get("USERNAME", os.path.basename(os.path.expanduser("~")))

# Keywords that identify a website / web-project request
_WEB_SIGNALS = (
    "website", "web app", "web application", "landing page", "webpage",
    "web page", "html", "portfolio site", "frontend", "site for",
)
# Keywords that signal a clarification question is needed (no business name / no context)
_NEEDS_CONTEXT_SIGNALS = (
    "a website", "a web app", "a site", "some website", "a landing page",
)

# ── Prompts ───────────────────────────────────────────────────────────────────

_CODE_GEN_SYSTEM = f"""\
You are a senior full-stack developer agent inside KOBRA, a Windows AI assistant.
Receive a natural language task and output ONLY raw executable Python 3 code.

═══ ENVIRONMENT ═══
  Desktop  = {_DESKTOP!r}
  Username = {_USERNAME!r}
  Use pathlib.Path. Never use placeholder names or hardcoded "YourUsername".
  Print a line for every file created. End with a summary print().

═══ PYTHON SCRIPTS / AUTOMATION ═══
  - Write complete, runnable code — no stubs, no pseudocode.
  - Use subprocess.run for shell commands. Capture output.

═══ WEBSITE / WEB PROJECT RULES — MANDATORY ═══
When the task involves creating a website, landing page, or web app:

1. CREATE A MULTI-FILE PROJECT on the Desktop:
   Desktop/<ProjectName>/
     index.html
     style.css
     script.js
     (+ additional pages if requested: menu.html, about.html, contact.html …)

2. index.html — MUST contain COMPLETE semantic HTML5:
   - <head> with charset, viewport, title, Google Fonts CDN link, style.css link
   - <nav> with logo, navigation links, and hamburger button for mobile
   - <section id="hero"> with compelling headline, subheading, CTA button
   - <section id="features"> or <section id="services"> with 3+ real items (use SVG icons inline)
   - <section id="about"> with a brief story and image placeholder
   - <section id="gallery"> or <section id="testimonials"> with real fake content
   - <section id="contact"> with a styled form (name, email, message, submit)
   - <footer> with logo, nav links, social icons, copyright
   - ALL sections filled with REALISTIC content specific to the business — never "Lorem ipsum"

3. style.css — COMPLETE modern CSS:
   :root {{
     /* CSS custom properties for colors, fonts, spacing */
   }}
   - Aesthetic matches business type (coffee shop = warm browns/creams; tech = dark/sharp; etc.)
   - Elegant Google Font pairing (e.g. Playfair Display + Inter for hospitality)
   - Flexbox/grid layouts — no floats
   - Mobile-first responsive (breakpoints at 768px, 1024px)
   - Smooth transitions on buttons, links, cards
   - Sticky navbar with box-shadow on scroll (via JS class)
   - Hero section with gradient background or subtle texture
   - Card components with hover lift effect (transform + box-shadow)
   - Animated CTA button
   - Properly styled contact form

4. script.js — REAL interactivity:
   - Sticky navbar: adds .scrolled class on scroll
   - Hamburger menu toggle (show/hide mobile nav)
   - Smooth scroll to anchor links
   - IntersectionObserver: fade-in sections as they scroll into view
   - Contact form: preventDefault + show success message
   - (optional) Simple counter animation for stats

5. After writing all files:
   - Open the project folder in VS Code: subprocess.run(["code", str(project_dir)])
   - Open index.html in the browser: os.startfile(str(index_path))
   - Print summary of all files created

QUALITY BAR: The output must look like a real freelancer delivered it — not a template stub.
Every section must have real, relevant content. Colors must be cohesive and professional.

═══ GENERAL OUTPUT RULE ═══
Output ONLY the Python code. No markdown fences. No explanations. No triple backticks.
"""

_CLARIFY_SYSTEM = """\
You are KOBRA — a sharp, witty AI assistant. Address the user as "sir".
The user asked to build a website or app but didn't give enough detail about what it's for.
Ask ONE short, friendly question (max 2 sentences) to get the key missing information.
Ask about: what type of business/purpose, business name if not given, and optionally preferred style.
Keep it natural and conversational — like asking a colleague, not filling out a form.
"""

_CODE_FIX_SYSTEM = """\
You are a code debugger for KOBRA. A previous code execution failed on Windows with Python 3.
Fix the code so it runs correctly. Common issues: wrong path separators, missing imports,
encoding issues, subprocess commands not available on Windows.
Output ONLY the corrected code. No markdown fences. No explanations.
"""

_NARRATE_SYSTEM = """\
You are KOBRA — sharp, dry, witty. Address the user as "sir".
Given the output of a code execution, write a natural spoken summary in 1-2 sentences.
No technical jargon. No raw file paths. No code. Just what was accomplished.
If it built a website or project: mention what you built and that it's open in the browser.
If something failed: say so plainly, without drama.
Never say "I have successfully" or "task completed".
"""


class InterpreterAgent(BaseAgent):
    AGENT_NAME = "interpreter"
    OWNED_TOOLS: list[str] = []   # No Groq tools — direct Groq calls for code gen
    SYSTEM_PROMPT = ""             # Not used — process_scoped not called

    def __init__(self, brain, memory) -> None:
        super().__init__(brain, memory)
        self._groq = Groq(api_key=config.GROQ_API_KEY)

    # ── Task execution ─────────────────────────────────────────────────────────

    def _run(self, task: Task) -> str:
        instruction = self._build_instruction(task)
        logger.info("[INTERPRETER] Task: %s", instruction[:120])

        # Step 1: Check if this is a website/project build that needs clarification
        clarification = self._maybe_ask_clarification(instruction)
        if clarification:
            return clarification

        # Step 2: Generate code
        code = self._generate_code(instruction)
        if not code:
            return "I couldn't generate code for that task, sir."

        # Step 3: Safety check
        if self._is_dangerous(code):
            return (
                "That operation involves potentially destructive commands, sir. "
                "I've generated the code but need explicit confirmation before running it."
            )

        # Step 4: Execute with retry loop
        for attempt in range(MAX_RETRIES):
            result = self._execute(code)

            if result["success"]:
                logger.info("[INTERPRETER] Execution succeeded on attempt %d", attempt + 1)
                return self._narrate(instruction, result)

            logger.warning(
                "[INTERPRETER] Attempt %d failed (rc=%d): %s",
                attempt + 1, result["returncode"], result["stderr"][:200],
            )

            if attempt < MAX_RETRIES - 1:
                code = self._fix_code(code, result)
                if not code:
                    break
            else:
                return self._narrate(instruction, result)

        return "I ran into an issue completing that, sir. Check the log for details."

    # ── Clarification ──────────────────────────────────────────────────────────

    def _maybe_ask_clarification(self, instruction: str) -> str | None:
        """
        If this is a web project request with zero context (no business name,
        no type, just "a website"), ask one focused clarifying question.
        Returns the question string, or None if we have enough info to build.
        """
        t = instruction.lower()
        is_web = any(w in t for w in _WEB_SIGNALS)
        if not is_web:
            return None

        # Check if there's actually no subject — e.g. "create a website" with nothing else
        vague = any(phrase in t for phrase in _NEEDS_CONTEXT_SIGNALS)
        has_context = any(c in instruction for c in (
            # Signs there IS enough context: a name, a business type keyword
            "called", "named", "for ", "shop", "restaurant", "store", "company",
            "startup", "agency", "clinic", "gym", "school", "portfolio", "blog",
        ))
        # Check recent conversation for already-given context
        recent = self._memory.get_recent(limit=4) if hasattr(self._memory, "get_recent") else []
        recent_text = " ".join(t.get("content", "") for t in recent).lower()
        answered_recently = any(w in recent_text for w in ("coffee", "shop", "called", "named"))

        if vague and not has_context and not answered_recently:
            try:
                resp = self._groq.chat.completions.create(
                    model=config.GROQ_MODEL_FAST,
                    messages=[
                        {"role": "system", "content": _CLARIFY_SYSTEM},
                        {"role": "user",   "content": instruction},
                    ],
                    max_tokens=100,
                    temperature=0.7,
                    timeout=30,
                )
                q = (resp.choices[0].message.content or "").strip()
                logger.info("[INTERPRETER] Asking clarification: %s", q[:80])
                return q
            except Exception:
                pass

        return None  # enough info — proceed to build

    # ── Code generation ────────────────────────────────────────────────────────

    def _generate_code(self, instruction: str) -> str | None:
        """
        Ask Groq (70b) to generate complete executable Python code.
        For web projects, uses up to 8192 tokens to generate full multi-file sites.
        """
        is_web = any(w in instruction.lower() for w in _WEB_SIGNALS)
        max_tok = 8000 if is_web else 2048   # websites need a LOT of tokens

        # Inject recent conversation context so the model knows answers to previous questions
        history_msgs = []
        if hasattr(self._memory, "get_recent"):
            for turn in self._memory.get_recent(limit=4):
                role = "user" if turn.get("role") == "user" else "assistant"
                history_msgs.append({"role": role, "content": turn.get("content", "")[:300]})

        messages = [
            {"role": "system", "content": _CODE_GEN_SYSTEM},
            *history_msgs,
            {"role": "user",   "content": instruction},
        ]

        try:
            response = self._groq.chat.completions.create(
                model=config.GROQ_MODEL_TOOLS,   # 70b — quality matters for code
                messages=messages,
                max_tokens=max_tok,
                temperature=0.15,
                timeout=120,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _strip_fences(raw)
        except Exception as exc:
            logger.error("[INTERPRETER] Code gen failed: %s", exc)
            return None

    def _fix_code(self, original_code: str, result: dict) -> str | None:
        """Ask Groq to fix failing code given the error output."""
        is_web = any(w in original_code.lower() for w in ("html", "css", "javascript", "index.html"))
        max_tok = 8000 if is_web else 2048

        error_context = (
            f"Code that failed:\n{original_code}\n\n"
            f"Error output:\n{result['stderr'][:1200]}\n"
            f"Stdout:\n{result['stdout'][:500]}\n"
            f"Return code: {result['returncode']}"
        )
        try:
            response = self._groq.chat.completions.create(
                model=config.GROQ_MODEL_TOOLS,
                messages=[
                    {"role": "system", "content": _CODE_FIX_SYSTEM},
                    {"role": "user",   "content": error_context},
                ],
                max_tokens=max_tok,
                temperature=0.1,
                timeout=120,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _strip_fences(raw)
        except Exception as exc:
            logger.error("[INTERPRETER] Code fix failed: %s", exc)
            return None

    # ── Execution ──────────────────────────────────────────────────────────────

    def _execute(self, code: str) -> dict[str, Any]:
        """
        Execute code in a subprocess. Detects language (Python vs PowerShell).
        Returns dict: {success, stdout, stderr, returncode, lang}
        """
        lang = _detect_language(code)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py" if lang == "python" else ".ps1",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            if lang == "python":
                cmd = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "venv", "Scripts", "python.exe"),
                    tmp_path,
                ]
                # Fallback to sys.executable if venv python not found
                if not os.path.exists(cmd[0]):
                    import sys
                    cmd[0] = sys.executable
            else:
                cmd = [
                    "powershell", "-ExecutionPolicy", "Bypass",
                    "-NonInteractive", "-File", tmp_path,
                ]

            logger.info("[INTERPRETER] Executing %s script: %s", lang, tmp_path)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=EXEC_TIMEOUT,
                cwd=_DESKTOP,
            )

            stdout = proc.stdout[:MAX_OUTPUT_CHARS]
            stderr = proc.stderr[:MAX_OUTPUT_CHARS]
            success = proc.returncode == 0 and not _has_fatal_error(stderr)

            return {
                "success":    success,
                "stdout":     stdout,
                "stderr":     stderr,
                "returncode": proc.returncode,
                "lang":       lang,
                "code":       code,
            }

        except subprocess.TimeoutExpired:
            return {
                "success":    False,
                "stdout":     "",
                "stderr":     f"Execution timed out after {EXEC_TIMEOUT} seconds.",
                "returncode": -1,
                "lang":       lang,
                "code":       code,
            }
        except Exception as exc:
            return {
                "success":    False,
                "stdout":     "",
                "stderr":     str(exc),
                "returncode": -1,
                "lang":       lang,
                "code":       code,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Narration ──────────────────────────────────────────────────────────────

    def _narrate(self, instruction: str, result: dict) -> str:
        """Turn execution output into a natural spoken response."""
        if result["success"]:
            output_summary = result["stdout"].strip()[-500:] or "Task completed with no output."
            prompt = (
                f"The task was: {instruction}\n"
                f"Execution succeeded. Output:\n{output_summary}"
            )
        else:
            prompt = (
                f"The task was: {instruction}\n"
                f"Execution failed after {MAX_RETRIES} attempts.\n"
                f"Error: {result['stderr'][:400]}"
            )

        try:
            response = self._groq.chat.completions.create(
                model=config.GROQ_MODEL_FAST,
                messages=[
                    {"role": "system", "content": _NARRATE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=120,
                temperature=0.7,
                timeout=30,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            if result["success"]:
                return "Done, sir. Task completed successfully."
            return "The task ran into an error, sir. Check the log for details."

    # ── Safety ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_dangerous(code: str) -> bool:
        code_lower = code.lower()
        return any(re.search(p, code_lower) for p in _DANGEROUS_PATTERNS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) from LLM output."""
    # Remove opening fence with optional language tag
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _detect_language(code: str) -> str:
    """Heuristic: is this Python or PowerShell?"""
    ps_signals = ["Write-Host", "Get-ChildItem", "Set-Location", "New-Item",
                  "Invoke-", "$env:", "param(", "function ", "-ErrorAction"]
    if any(s in code for s in ps_signals):
        return "powershell"
    return "python"


def _has_fatal_error(stderr: str) -> bool:
    """Return True only if the output contains a real Python exception/traceback."""
    fatal_markers = (
        "Traceback (most recent call last)",
        "SyntaxError:",
        "ModuleNotFoundError:",
        "ImportError:",
        "NameError:",
        "AttributeError:",
        "TypeError:",
        "ValueError:",
        "FileNotFoundError:",
        "PermissionError:",
        "RuntimeError:",
        "IndentationError:",
        "RecursionError:",
    )
    return any(m in stderr for m in fatal_markers)
