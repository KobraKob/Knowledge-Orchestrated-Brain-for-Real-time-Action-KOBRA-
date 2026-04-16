"""
agents/screen_agent.py — Vision-driven screen navigation agent for KOBRA.

Agentic loop:
  1. Take screenshot + ask Groq vision what's on screen
  2. Decide next action (click, type, scroll, etc.)
  3. Execute, verify, repeat up to MAX_STEPS

Trigger phrases (task_router.py):
  "what's on my screen", "click the [element]", "read the error",
  "what does this code do", "fill in [form]"
"""

import logging

from agents.base_agent import BaseAgent
from models import Task
from tools.screen import take_screenshot, _send_to_vision

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are KOBRA's screen navigation module. You control the computer visually.

Rules:
- ALWAYS read_screen first to understand the current state before acting.
- Use click_element to interact with UI elements by description — never guess coordinates.
- After clicking, take another screenshot to verify the action worked.
- If a task seems done, say "Task complete" to end the loop.
- Max 8 steps per task — if still not done, report what you've accomplished.
- Be efficient. Don't narrate every step — just act.

Available tools: read_screen, click_element, type_text, press_hotkey, scroll_screen, take_screenshot
"""


class ScreenAgent(BaseAgent):
    AGENT_NAME = "screen"
    OWNED_TOOLS = [
        "read_screen", "click_element",
        "take_screenshot", "type_text", "press_hotkey", "scroll_screen",
        "speak_only",
    ]
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    MAX_STEPS = 8
    _DONE_PHRASES = (
        "task complete", "done", "finished", "accomplished",
        "completed", "all done", "that's it",
    )

    def _run(self, task: Task) -> str:
        instruction = self._build_instruction(task)

        for step in range(self.MAX_STEPS):
            # Take screenshot and ask vision for current state
            screenshot_path = take_screenshot()
            vision_context = _send_to_vision(
                screenshot_path,
                f"I am trying to: {instruction}\n"
                f"Step {step + 1}: What is currently on screen? "
                f"What should I do next to accomplish the task?"
            )

            # Let Groq decide the next action based on vision context
            action_instruction = (
                f"Task: {instruction}\n"
                f"Current screen state: {vision_context}\n"
                f"Step {step + 1}/{self.MAX_STEPS}: What is the single next action to take? "
                f"Use a tool to execute it. If the task is complete, call speak_only."
            )

            result = self._brain.process_scoped(
                instruction=action_instruction,
                tool_names=self.OWNED_TOOLS,
                system_prompt=self.SYSTEM_PROMPT,
            )

            # Check for task completion
            if any(phrase in result.lower() for phrase in self._DONE_PHRASES):
                return result

            logger.debug("[SCREEN] Step %d result: %.80s", step + 1, result)

        return f"Completed {self.MAX_STEPS} steps on: {instruction}"
