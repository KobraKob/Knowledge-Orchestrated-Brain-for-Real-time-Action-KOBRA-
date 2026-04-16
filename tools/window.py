"""
tools/window.py — Window management and Focus Mode for KOBRA.

Provides voice-controllable workspace layouts:
  - snap/focus/close individual windows
  - activate named focus mode presets (coding, gaming, research, break)
  - create custom modes from voice description
"""

import logging
import os
import subprocess
import time

import config

logger = logging.getLogger(__name__)


def _set_volume(level: int) -> None:
    """Set system volume 0-100 using PowerShell."""
    try:
        subprocess.run(
            ["powershell", "-Command",
             f"$obj = New-Object -ComObject WScript.Shell; "
             f"for ($i=0; $i -lt 50; $i++) {{$obj.SendKeys([char]174)}}; "
             f"$steps = [math]::Round({level}/2); "
             f"for ($i=0; $i -lt $steps; $i++) {{$obj.SendKeys([char]175)}}"],
            capture_output=True, timeout=5
        )
    except Exception:
        pass


# ── Snap key map for pyautogui ────────────────────────────────────────────────

_SNAP_HOTKEYS: dict[str, tuple[str, ...]] = {
    "left":     ("win", "left"),
    "right":    ("win", "right"),
    "maximize": ("win", "up"),
    "minimize": ("win", "down"),
}


# ── Low-level window helpers ──────────────────────────────────────────────────

def get_open_windows() -> list[dict]:
    """Return visible windows as list of {title, handle, rect} dicts."""
    try:
        from pywinauto import Desktop
        windows = []
        for w in Desktop(backend="uia").windows():
            try:
                if w.is_visible() and w.window_text():
                    windows.append({
                        "title":  w.window_text(),
                        "handle": w.handle,
                        "rect":   w.rectangle(),
                    })
            except Exception:
                pass
        return windows
    except Exception as exc:
        logger.error("get_open_windows failed: %s", exc)
        return []


def focus_window(app_name: str) -> str:
    """Bring the first window whose title contains app_name to the foreground."""
    try:
        import pywinauto
        matches = [w for w in get_open_windows()
                   if app_name.lower() in w["title"].lower()]
        if not matches:
            return f"No window matching '{app_name}' found."
        win = pywinauto.Application().connect(handle=matches[0]["handle"])
        win.top_window().set_focus()
        return f"Focused {matches[0]['title']}."
    except Exception as exc:
        logger.error("focus_window failed: %s", exc)
        return f"Could not focus '{app_name}': {exc}"


def snap_window(app_name: str, position: str) -> str:
    """Snap a window to a screen position using Windows keyboard shortcuts."""
    result = focus_window(app_name)
    if "No window" in result or "Could not" in result:
        return result
    try:
        import pyautogui
        keys = _SNAP_HOTKEYS.get(position.lower(), ("win", "up"))
        pyautogui.hotkey(*keys)
        time.sleep(0.2)
        return f"Snapped {app_name} to {position}."
    except Exception as exc:
        logger.error("snap_window failed: %s", exc)
        return f"Could not snap '{app_name}': {exc}"


def close_window(app_name: str) -> str:
    """Close the first window whose title contains app_name."""
    try:
        import pywinauto
        matches = [w for w in get_open_windows()
                   if app_name.lower() in w["title"].lower()]
        if not matches:
            return f"No window matching '{app_name}' found."
        win = pywinauto.Application().connect(handle=matches[0]["handle"])
        win.top_window().close()
        return f"Closed {matches[0]['title']}."
    except Exception as exc:
        logger.error("close_window failed: %s", exc)
        return f"Could not close '{app_name}': {exc}"


def switch_virtual_desktop(number: int) -> str:
    """Switch to virtual desktop N using Win+Ctrl+Right (cycles forward)."""
    try:
        import pyautogui
        target = max(1, int(number))
        for _ in range(target - 1):
            pyautogui.hotkey("ctrl", "win", "right")
            time.sleep(0.3)
        return f"Switched to desktop {number}."
    except Exception as exc:
        return f"Could not switch desktop: {exc}"


# ── Focus Mode presets ────────────────────────────────────────────────────────

def activate_focus_mode(mode_name: str) -> str:
    """
    Activate a named workspace layout from config.FOCUS_MODES.
    Opens apps, snaps windows, closes distractions, plays music.
    """
    mode = config.FOCUS_MODES.get(mode_name.lower())
    if not mode:
        available = ", ".join(config.FOCUS_MODES.keys())
        return f"No focus mode named '{mode_name}'. Available: {available}."

    results: list[str] = []

    # 1. Open apps
    for app in mode.get("open", []):
        try:
            from tools.system import open_app
            open_app(app)
            time.sleep(0.8)
        except Exception as exc:
            logger.warning("Could not open %s: %s", app, exc)

    # 2. Snap windows
    for app, pos in mode.get("snap", {}).items():
        r = snap_window(app, pos)
        results.append(r)
        time.sleep(0.3)

    # 3. Close distractions
    for app in mode.get("close", []):
        close_window(app)

    # 4. Play music
    if "play" in mode:
        try:
            from tools.media import play_media
            play_media(mode["play"])
        except Exception as exc:
            logger.warning("Could not play music: %s", exc)

    # 5. Set volume
    if "volume" in mode:
        try:
            _set_volume(mode["volume"])
        except Exception as exc:
            logger.warning("Could not set volume: %s", exc)

    return mode.get("speak", f"{mode_name} mode activated.")


# Alias used by TOOL_REGISTRY and brain.py tool definitions
focus_mode = activate_focus_mode


def list_focus_modes() -> str:
    """Return a spoken list of available focus mode names."""
    modes = list(config.FOCUS_MODES.keys())
    if not modes:
        return "No focus modes configured."
    return "Available modes: " + ", ".join(modes) + "."
