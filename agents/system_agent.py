"""
agents/system_agent.py — Desktop/system control agent.

Handles: open_app, install_app, run_command, get_system_info, create_folder,
         type_text, press_hotkey, control_volume, get_clipboard, set_clipboard,
         click_at, double_click_at, right_click_at, move_mouse, scroll_screen,
         take_screenshot, speak_only.
"""

from agents.base_agent import BaseAgent
from models import Task


class SystemAgent(BaseAgent):
    AGENT_NAME = "system"
    OWNED_TOOLS = [
        # App control
        "open_app",
        "install_app",
        "run_command",
        "get_system_info",
        "create_folder",
        # Keyboard / audio
        "type_text",
        "press_hotkey",
        "control_volume",
        "get_clipboard",
        "set_clipboard",
        # Mouse / screen
        "click_at",
        "double_click_at",
        "right_click_at",
        "move_mouse",
        "scroll_screen",
        "take_screenshot",
        # Fallback
        "speak_only",
    ]
    SYSTEM_PROMPT = (
        "You are KOBRA's system control agent with FULL access to the PC. "
        "Address the user as 'sir'. Be direct. Never ask for confirmation on non-destructive tasks. "
        "Return results as short strings suitable for text-to-speech.\n\n"
        "AVAILABLE TOOLS — use ONLY these exact names, no others:\n"
        "  open_app(app_name)             — open any app (UWP, Store, .exe)\n"
        "  install_app(app_name)          — install via winget (Spotify, Chrome, etc.)\n"
        "  run_command(command)           — run any shell command, full access\n"
        "  get_system_info(info_type)     — time/date/battery/cpu/ram/all\n"
        "  create_folder(path)            — create a directory\n"
        "  type_text(text)                — type text at cursor\n"
        "  press_hotkey(keys)             — e.g. 'ctrl+c', 'win+d', 'alt+f4'\n"
        "  control_volume(action, steps)  — up/down/mute/unmute\n"
        "  get_clipboard()                — read clipboard\n"
        "  set_clipboard(text)            — write to clipboard\n"
        "  click_at(x, y)                 — left-click at screen coordinates\n"
        "  double_click_at(x, y)          — double-click at coordinates\n"
        "  right_click_at(x, y)           — right-click at coordinates\n"
        "  move_mouse(x, y)               — move cursor to coordinates\n"
        "  scroll_screen(direction, amount) — scroll up or down\n"
        "  take_screenshot()              — capture screen, returns file path\n"
        "  speak_only(response)           — speak a response with no action\n\n"
        "RULES:\n"
        "- To install software: use install_app, NOT run_command.\n"
        "- To open Microsoft Store: use open_app('Microsoft Store').\n"
        "- To download & install any app: use install_app('AppName').\n"
        "- To click UI buttons after opening an app: use take_screenshot() first, "
        "then click_at(x, y) at the button's coordinates.\n"
        "- NEVER call tools that are not in the list above."
    )

    def _run(self, task: Task) -> str:
        return self._brain.process_scoped(
            self._build_instruction(task),
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )
