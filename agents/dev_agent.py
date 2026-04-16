"""
agents/dev_agent.py — Developer workflow agent with Code Assistant Mode.

Handles: create_file, open_vscode, scaffold_project, run_command, create_folder.

Code Assistant Mode activates when VS Code is the foreground window:
  - "What does this code do?" → read_screen + vision explanation
  - "Fix the error" → read terminal region + Groq fix
  - "Commit this" → git diff → Groq message → git commit
  - "Run the tests" → pytest → parse → speak results

ReAct Loop (v4):
  Complex multi-step instructions use Observe → Think → Act → Observe → ... → Done
  Simple single-tool instructions skip ReAct for low overhead.
"""

import json
import logging
import os
import re

import config
from agents.base_agent import BaseAgent
from groq import Groq
from models import Task

logger = logging.getLogger(__name__)

_MAX_REACT_STEPS = 5

_REACT_SYSTEM = """\
You are KOBRA's developer agent. You have tools available and must complete the task step by step.

At each step, respond with ONLY a JSON object (no markdown fences, no extra text):
{
  "thought": "What I know so far and what I should do next",
  "action": "tool_name",
  "action_input": "input to the tool",
  "final_answer": ""
}

When the task is complete, set action to "done" and fill final_answer with a short spoken confirmation (1-2 sentences, address user as sir).

Available tools:
- create_file: Create a file. action_input format: "path|||content" (pipe-separated)
- read_file: Read a file's content. action_input: file path
- run_command: Run a Windows shell/PowerShell command. action_input: the command string
- open_vscode: Open VS Code. action_input: path to open (or empty for current dir)
- done: Task complete. Fill final_answer with spoken result.

Rules:
- Always think before acting
- If a tool fails, try a different approach
- Never repeat the exact same action twice
- Be concise and direct in final_answer — KOBRA will speak this to the user
- Use full Windows paths (e.g. C:\\Users\\username\\Desktop\\file.txt)
"""


class DevAgent(BaseAgent):
    AGENT_NAME = "dev"
    OWNED_TOOLS = [
        "create_file",
        "open_vscode",
        "scaffold_project",
        "run_command",
        "create_folder",
        "read_screen",
        "speak_only",
    ]

    # Desktop path resolved at class definition — agents use this for file paths
    _DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

    SYSTEM_PROMPT = (
        "You are KOBRA's developer agent with full filesystem access. "
        "Address the user as 'sir'. Be direct. Execute immediately — never ask for confirmation. "
        "Return a short confirmation string (1 sentence) for text-to-speech.\n\n"
        f"IMPORTANT: Windows Desktop is at: {os.path.join(os.path.expanduser('~'), 'Desktop')}\n"
        "When the user says 'desktop', use that exact path.\n\n"
        "AVAILABLE TOOLS — use ONLY these exact names, spelled exactly as shown:\n"
        "  create_file(path, content)     — create a file with content at a Windows path\n"
        "  create_folder(path)            — create a directory\n"
        "  open_vscode(path)              — open VS Code at a path\n"
        "  scaffold_project(project_name, project_type, location) — scaffold a new project\n"
        "  run_command(command)           — run Windows shell (cmd/PowerShell) commands ONLY\n"
        "  read_screen(question, region)  — take screenshot and understand what's on screen\n"
        "  speak_only(response)           — speak a response with no file action\n\n"
        "FILE CREATION RULES — READ CAREFULLY:\n"
        "- To create or write a file: ALWAYS use create_file(path, content). Never use anything else.\n"
        "- create_file IS a tool — call it directly. NEVER call it via run_command.\n"
        "- run_command is for: git, pip, npm, pytest, mkdir, and other shell operations ONLY.\n"
        "- NEVER run 'create_file ...' or 'echo > ...' or 'ls > ...' as shell commands.\n"
        "- When asked to 'create a file AND fill it with content': do BOTH in ONE create_file call.\n"
        "  Generate the full content yourself and put it in the content parameter.\n"
        "- If context provides web search results or facts, include them in the content.\n"
        "- If no context is provided, use your own knowledge to generate relevant content.\n\n"
        "WINDOWS RULES:\n"
        "- Use full Windows paths: C:\\\\Users\\\\{username}\\\\Desktop\\\\filename.txt\n"
        "- NEVER use Unix paths (/home/user) or Linux commands (ls, echo -e, cat)\n"
        "- NEVER use placeholder paths like YourUsername — use the real username\n"
        f"- Real username: {os.environ.get('USERNAME', 'user')}\n\n"
        "[CODE ASSISTANT MODE]\n"
        "When VS Code is the active window:\n"
        "- Use read_screen to see what's visible before explaining code\n"
        "- Use run_command for git, pytest, pip operations\n"
        "- Be precise with code — this is not the time for wit"
    )

    def __init__(self, brain, memory) -> None:
        super().__init__(brain, memory)
        self._client = Groq(api_key=config.GROQ_API_KEY)

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def _react_loop(
        self,
        task_instruction: str,
        abort_flag,
    ) -> str:
        """
        ReAct loop: Observe → Think → Act → Observe → ... → Done
        Returns the final answer string.
        """
        tools = self._make_dev_tools()
        messages = [
            {"role": "system", "content": _REACT_SYSTEM},
            {"role": "user", "content": f"Task: {task_instruction}"},
        ]

        for step in range(_MAX_REACT_STEPS):
            if abort_flag.is_set():
                return "Aborted, sir."

            try:
                response = self._client.chat.completions.create(
                    model=config.GROQ_MODEL_TOOLS,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                    timeout=30,
                )
                raw = (response.choices[0].message.content or "").strip()
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                step_data = json.loads(clean)
            except Exception as exc:
                logger.warning("[DEV REACT] Step %d parse failed: %s", step, exc)
                break

            thought = step_data.get("thought", "")
            action = step_data.get("action", "done")
            action_input = step_data.get("action_input", "")
            final_answer = step_data.get("final_answer", "")

            logger.info(
                "[DEV REACT] Step %d — thought: %s | action: %s",
                step, thought[:60], action,
            )

            # Append assistant reasoning to messages
            messages.append({"role": "assistant", "content": raw})

            if action == "done" or step == _MAX_REACT_STEPS - 1:
                return final_answer or "Done, sir."

            # Execute the chosen tool
            tool_fn = tools.get(action)
            if not tool_fn:
                observation = (
                    f"Error: Tool '{action}' not found. "
                    f"Available: {list(tools.keys())}"
                )
            else:
                try:
                    result = tool_fn(action_input)
                    # Handle both string and ToolResult-like return types
                    if hasattr(result, "output"):
                        observation = result.output if result.success else f"Tool failed: {result.output}"
                    else:
                        observation = str(result)
                except Exception as exc:
                    observation = f"Tool error: {exc}"

            logger.info("[DEV REACT] Step %d observation: %.100s", step, observation)
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "I completed the task, sir."

    def _make_dev_tools(self) -> dict:
        """Build the tool callables used by the ReAct loop."""

        def create_file(action_input: str) -> str:
            """action_input: 'path|||content'"""
            try:
                from tools.system import create_file as _create_file  # noqa: F401
            except ImportError:
                pass
            if "|||" in action_input:
                path, content = action_input.split("|||", 1)
            else:
                # Try to split on first newline — path on first line, rest is content
                parts = action_input.split("\n", 1)
                path = parts[0].strip()
                content = parts[1] if len(parts) > 1 else ""
            path = path.strip()
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"File created: {path}"
            except Exception as exc:
                return f"Failed to create file: {exc}"

        def read_file(action_input: str) -> str:
            path = action_input.strip()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()[:3000]
            except Exception as exc:
                return f"Failed to read file: {exc}"

        def run_command(action_input: str) -> str:
            try:
                from tools.system import run_command as _run_cmd
                result = _run_cmd(action_input.strip())
                return str(result)[:2000]
            except Exception as exc:
                return f"Command error: {exc}"

        def open_vscode(action_input: str) -> str:
            try:
                path = action_input.strip() or "."
                self._brain._dispatch_tool("open_vscode", {"path": path})
                return f"VS Code opened at {path}"
            except Exception as exc:
                return f"Failed to open VS Code: {exc}"

        return {
            "create_file": create_file,
            "read_file": read_file,
            "run_command": run_command,
            "open_vscode": open_vscode,
        }

    # ── Complexity check ───────────────────────────────────────────────────────

    @staticmethod
    def _is_complex(instruction: str) -> bool:
        """
        Returns True if the instruction warrants the ReAct multi-step loop.
        Heuristic: more than 10 words OR contains coordinating keywords.
        """
        words = instruction.split()
        if len(words) > 10:
            return True
        lower = instruction.lower()
        return any(kw in lower for kw in (" and ", " then ", " also ", " after ", " next "))

    # ── Code Assistant Mode helpers ────────────────────────────────────────────

    def explain_visible_code(self) -> str:
        """Read the visible code area and explain it."""
        from tools.screen import read_screen
        return read_screen(
            "Describe the code visible on screen. What does it do? Any obvious issues?",
            region="code",
        )

    def fix_terminal_error(self) -> str:
        """Read the terminal area, identify the error, and generate a fix."""
        from tools.screen import read_screen
        error_desc = read_screen(
            "Read the terminal error message visible on screen. "
            "What is the error and what is the fix?",
            region="terminal",
        )
        fix = self._brain.process_conversational(
            f"Given this error:\n{error_desc}\n\nWrite the corrected code or command."
        )
        return fix

    def generate_commit_message(self) -> str:
        """Run git diff, generate a commit message, and commit."""
        from tools.system import run_command
        diff = run_command("git diff --staged")
        if not diff.strip() or "nothing to commit" in diff.lower():
            diff = run_command("git diff HEAD")
        if not diff.strip():
            return "Nothing to commit, sir."
        prompt = f"Write a concise git commit message (one line) for these changes:\n{diff[:2000]}"
        message = self._brain.process_conversational(prompt).strip().strip('"\'')
        # Remove any markdown fences the model might add
        message = re.sub(r'^```.*\n?', '', message).strip()
        result = run_command(f'git commit -m "{message}"')
        if "error" in result.lower() or "fatal" in result.lower():
            return f"Commit failed, sir: {result[:100]}"
        return f"Committed: {message}"

    def run_tests_and_report(self) -> str:
        """Run pytest and speak a summary of results."""
        from tools.system import run_command
        result = run_command("python -m pytest --tb=short -q 2>&1")
        lines = result.split("\n")
        failed = [l for l in lines if "FAILED" in l or "ERROR" in l]
        passed = [l for l in lines if " passed" in l]
        if not failed:
            summary = passed[-1].strip() if passed else "All good."
            return f"All tests passed, sir. {summary}"
        summary = "; ".join(f.strip() for f in failed[:3])
        return f"Tests failed, sir. {len(failed)} failure(s): {summary}"

    # ── Main entry ─────────────────────────────────────────────────────────────

    def _run(self, task: Task) -> str:
        instruction = self._build_instruction(task)
        instruction_lower = instruction.lower()

        # Code Assistant Mode — route to specialized handlers (bypass ReAct)
        if any(w in instruction_lower for w in ("what does this code", "explain this", "what's on screen",
                                                  "explain the code", "what is this code")):
            return self.explain_visible_code()

        if any(w in instruction_lower for w in ("fix the error", "fix the terminal", "what's the error",
                                                  "read the error")):
            return self.fix_terminal_error()

        if any(w in instruction_lower for w in ("commit this", "commit my changes", "generate commit",
                                                  "make a commit")):
            return self.generate_commit_message()

        if any(w in instruction_lower for w in ("run the tests", "run tests", "run pytest",
                                                  "run the test suite")):
            return self.run_tests_and_report()

        # ReAct loop for complex multi-step instructions
        if self._is_complex(instruction):
            logger.info("[DEV] Using ReAct loop for complex instruction: %.80s", instruction)
            # Pass a no-op abort flag if called from _run (abort_flag lives in execute())
            import threading
            _dummy_flag = threading.Event()
            return self._react_loop(instruction, _dummy_flag)

        # Simple single-action instruction — direct brain dispatch (low overhead)
        return self._brain.process_scoped(
            instruction,
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )

    def execute(self, task, abort_flag) -> "TaskResult":
        """Override execute to pass abort_flag into the ReAct loop."""
        import time
        from models import TaskResult

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
            instruction = self._build_instruction(task)
            instruction_lower = instruction.lower()

            # Code Assistant Mode handlers (no ReAct)
            if any(w in instruction_lower for w in ("what does this code", "explain this",
                                                      "what's on screen", "explain the code",
                                                      "what is this code")):
                output = self.explain_visible_code()

            elif any(w in instruction_lower for w in ("fix the error", "fix the terminal",
                                                        "what's the error", "read the error")):
                output = self.fix_terminal_error()

            elif any(w in instruction_lower for w in ("commit this", "commit my changes",
                                                        "generate commit", "make a commit")):
                output = self.generate_commit_message()

            elif any(w in instruction_lower for w in ("run the tests", "run tests", "run pytest",
                                                        "run the test suite")):
                output = self.run_tests_and_report()

            # ReAct loop for complex instructions
            elif self._is_complex(instruction):
                logger.info("[DEV] Using ReAct loop for complex instruction: %.80s", instruction)
                output = self._react_loop(instruction, abort_flag)

            else:
                # Simple direct dispatch
                output = self._brain.process_scoped(
                    instruction,
                    self.OWNED_TOOLS,
                    self.SYSTEM_PROMPT,
                )

            success = True

        except Exception as exc:
            logger.exception("[%s] Task %s raised: %s", self.AGENT_NAME, task.id, exc)
            output = f"Error: {exc}"
            success = False

        duration = time.perf_counter() - start
        logger.info("[%s] Task %s done in %.2fs", self.AGENT_NAME, task.id, duration)

        from models import TaskResult
        return TaskResult(
            task_id=task.id,
            agent_name=self.AGENT_NAME,
            success=success,
            output=output,
            duration_seconds=duration,
        )
