"""
agents/browser_agent.py — Playwright browser automation agent for KOBRA.

Handles services with no public API:
  - WhatsApp Web (send_message, read_messages)

Uses a persistent Chromium context (kobra_browser_session/) so WhatsApp
login survives restarts. First run shows QR code (headless=False); after
scanning, set BROWSER_HEADLESS=True in config for silent operation.

The browser is started lazily on first task — no startup cost if unused.
Cleanup is called by main.py on shutdown.

ReAct Loop (v4):
  Browser tasks are inherently multi-step — always uses the ReAct loop to
  decide which browser tool to call at each sub-step.
"""

import json
import logging
import re
import threading

import config
from agents.base_agent import BaseAgent
from contact_store import ContactStore, ContactNotFoundError
from credential_store import CredentialStore
from groq import Groq
from integrations.base_integration import NotAuthenticatedError, IntegrationError
from integrations.whatsapp import WhatsAppIntegration
from models import Task, TaskResult

logger = logging.getLogger(__name__)

_MAX_REACT_STEPS = 5

_REACT_SYSTEM = """\
You are KOBRA's browser agent. You control web applications through a browser.
Your primary role is WhatsApp Web automation.

At each step, respond with ONLY a JSON object (no markdown fences, no extra text):
{
  "thought": "What I know so far and what I should do next",
  "action": "tool_name",
  "action_input": "input to the tool",
  "final_answer": ""
}

When the task is complete, set action to "done" and fill final_answer with a short spoken confirmation.

Available tools:
- whatsapp_send_message: Send a WhatsApp message. action_input format: "contact_name|||message text"
- whatsapp_read_messages: Read recent WhatsApp messages. action_input format: "contact_name|||count" (count is optional, default 5)
- done: Task complete. Fill final_answer with spoken result (max 2 sentences, no markdown).

Rules:
- NEVER fabricate message content. Use exactly what sir asked to send.
- If a contact is unknown, report it clearly — do not guess.
- Output is spoken aloud. No markdown, no bullets, no URLs. Max 2 sentences.
- Do not attempt tasks outside the available tools.
- Always think before acting.
- If a tool fails, try a different approach or report the error.
- Never repeat the exact same action twice.
"""


class BrowserAgent(BaseAgent):
    AGENT_NAME = "browser"
    OWNED_TOOLS = [
        "whatsapp_send_message",
        "whatsapp_read_messages",
        "speak_only",
    ]

    SYSTEM_PROMPT = """\
You are KOBRA's browser agent. You control web applications through a browser.
Your primary role is WhatsApp Web automation.

RULES:
- For WhatsApp messages: use whatsapp_send_message.
- For reading WhatsApp: use whatsapp_read_messages.
- NEVER fabricate message content. Use exactly what sir asked to send.
- If a contact is unknown: report it clearly — do not guess.
- Output is spoken aloud. No markdown, no bullets, no URLs. Max 2 sentences.
- Do not attempt tasks outside the available tools.
"""

    def __init__(
        self,
        brain,
        memory,
        credential_store: CredentialStore,
        contact_store: ContactStore,
    ) -> None:
        super().__init__(brain, memory)
        self._creds = credential_store
        self._contacts = contact_store
        self._whatsapp = WhatsAppIntegration(credential_store, contact_store)
        self._playwright = None
        self._browser = None
        self._page = None
        self._browser_lock = threading.Lock()
        self._started = False
        self._client = Groq(api_key=config.GROQ_API_KEY)

    # ── Browser lifecycle ──────────────────────────────────────────────────────

    def _ensure_browser(self) -> bool:
        """Start the Playwright browser if not already running. Thread-safe."""
        if self._started and self._page is not None:
            return True

        with self._browser_lock:
            if self._started and self._page is not None:
                return True

            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                logger.error(
                    "[BROWSER] playwright not installed. "
                    "Run: pip install playwright && playwright install chromium"
                )
                return False

            try:
                session_path = getattr(config, "BROWSER_SESSION_PATH", "kobra_browser_session")
                headless = getattr(config, "BROWSER_HEADLESS", False)

                logger.info("[BROWSER] Starting Chromium (headless=%s, session=%s)",
                            headless, session_path)

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch_persistent_context(
                    session_path,
                    headless=headless,
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )

                # Get or create a page
                pages = self._browser.pages
                self._page = pages[0] if pages else self._browser.new_page()
                self._whatsapp.set_page(self._page)
                self._started = True
                logger.info("[BROWSER] Browser context ready.")
                return True

            except Exception as exc:
                logger.error("[BROWSER] Failed to start browser: %s", exc)
                return False

    def cleanup(self) -> None:
        """Gracefully close the Playwright browser. Call on KOBRA shutdown."""
        if self._browser is not None:
            try:
                self._browser.close()
                logger.info("[BROWSER] Browser closed.")
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._started = False
        self._page = None

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def _react_loop(
        self,
        task_instruction: str,
        abort_flag: threading.Event,
    ) -> str:
        """
        ReAct loop: Observe → Think → Act → Observe → ... → Done
        Returns the final answer string.
        """
        tools = self._make_react_tools()
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
                logger.warning("[BROWSER REACT] Step %d parse failed: %s", step, exc)
                break

            thought = step_data.get("thought", "")
            action = step_data.get("action", "done")
            action_input = step_data.get("action_input", "")
            final_answer = step_data.get("final_answer", "")

            logger.info(
                "[BROWSER REACT] Step %d — thought: %s | action: %s",
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
                except ContactNotFoundError as exc:
                    observation = (
                        f"Contact not found: {exc.name!r}. "
                        "Ask sir for their number to save it."
                    )
                except NotAuthenticatedError as exc:
                    observation = (
                        f"Not authenticated with {exc.service}. "
                        "QR code scan required."
                    )
                except IntegrationError as exc:
                    observation = f"Integration error: {exc}"
                except Exception as exc:
                    observation = f"Tool error: {exc}"

            logger.info("[BROWSER REACT] Step %d observation: %.100s", step, observation)
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "I completed the task, sir."

    def _make_react_tools(self) -> dict:
        """Build the tool callables used by the ReAct loop."""
        wa = self._whatsapp

        def whatsapp_send_message(action_input: str) -> str:
            """action_input: 'contact_name|||message text'"""
            if "|||" in action_input:
                to_name, message = action_input.split("|||", 1)
            else:
                # Fallback: try to split on first comma
                parts = action_input.split(",", 1)
                to_name = parts[0].strip()
                message = parts[1].strip() if len(parts) > 1 else action_input
            return wa.send_message(to_name.strip(), message.strip())

        def whatsapp_read_messages(action_input: str) -> str:
            """action_input: 'contact_name' or 'contact_name|||count'"""
            if "|||" in action_input:
                from_name, count_str = action_input.split("|||", 1)
                try:
                    count = int(count_str.strip())
                except ValueError:
                    count = 5
            else:
                from_name = action_input
                count = 5
            return wa.read_messages(from_name.strip(), count=count)

        return {
            "whatsapp_send_message": whatsapp_send_message,
            "whatsapp_read_messages": whatsapp_read_messages,
        }

    # ── Task execution ─────────────────────────────────────────────────────────

    def _run(self, task: Task) -> str:
        # _run is called by base execute() which doesn't pass abort_flag.
        # We delegate to execute() override for proper abort support.
        # This path is a safety fallback.
        if not self._ensure_browser():
            return (
                "Browser automation is unavailable, sir. "
                "Make sure Playwright is installed: pip install playwright && playwright install chromium"
            )
        instruction = self._build_instruction(task)
        import threading as _threading
        dummy_flag = _threading.Event()
        return self._react_loop(instruction, dummy_flag)

    def execute(self, task: Task, abort_flag: threading.Event) -> TaskResult:
        """Override execute to pass abort_flag into the ReAct loop and handle browser errors."""
        import time

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

        if not self._ensure_browser():
            return TaskResult(
                task_id=task.id,
                agent_name=self.AGENT_NAME,
                success=False,
                output=(
                    "Browser automation is unavailable, sir. "
                    "Make sure Playwright is installed: "
                    "pip install playwright && playwright install chromium"
                ),
                duration_seconds=time.perf_counter() - start,
            )

        try:
            instruction = self._build_instruction(task)
            output = self._react_loop(instruction, abort_flag)
            success = True
        except NotAuthenticatedError as exc:
            output = (
                f"I need to log into {exc.service} first, sir. "
                "Please scan the QR code on screen."
            )
            success = False
        except ContactNotFoundError as exc:
            output = (
                f"I don't have a WhatsApp number for {exc.name!r}, sir. "
                "Tell me their number and I'll save it."
            )
            success = False
        except IntegrationError as exc:
            output = f"Browser automation error, sir: {exc}"
            success = False
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
