"""
tools/system.py — System-level tools for KOBRA.

open_app        — launch an application by name
run_command     — run a whitelisted shell command
get_system_info — CPU, RAM, battery, time, date
create_folder   — create a directory
type_text       — type text at the current cursor position
press_hotkey    — send a keyboard shortcut
control_volume  — raise / lower / mute system volume
get_clipboard   — read clipboard content
set_clipboard   — write to clipboard
"""

import logging
import os
import platform
import subprocess
from datetime import datetime

import psutil

import config

logger = logging.getLogger(__name__)


# ── App launching ──────────────────────────────────────────────────────────────

def open_app(app_name: str) -> str:
    """
    Try to launch an application by name.
    Strategy: direct Popen → walk common install dirs → os.startfile fallback.
    """
    logger.info("[TOOL] open_app: %s", app_name)

    # 1. Direct launch (works if the app is in PATH)
    try:
        subprocess.Popen(
            [app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0,
        )
        return f"Opened {app_name}."
    except (FileNotFoundError, OSError):
        pass

    # 2. Walk common Windows install directories
    search_dirs = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        os.environ.get("APPDATA", ""),
    ]
    name_lower = app_name.lower()
    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fname in files:
                if fname.lower().endswith(".exe") and name_lower in fname.lower():
                    exe_path = os.path.join(root, fname)
                    try:
                        subprocess.Popen(
                            [exe_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=subprocess.DETACHED_PROCESS,
                        )
                        return f"Opened {app_name}."
                    except OSError:
                        pass

    # 3. os.startfile — Windows file-association fallback
    try:
        os.startfile(app_name)
        return f"Opened {app_name}."
    except OSError:
        pass

    return f"Couldn't find '{app_name}' on your system, sir."


# ── Shell commands ─────────────────────────────────────────────────────────────

def run_command(command: str) -> str:
    """
    Run a shell command if it starts with a whitelisted prefix.
    Output is capped at 500 chars to avoid flooding TTS.
    """
    logger.info("[TOOL] run_command: %s", command)

    stripped = command.strip()
    first_token = stripped.split()[0].lower() if stripped else ""

    if not any(first_token == w.lower() for w in config.COMMAND_WHITELIST):
        return (
            f"Command not permitted: '{stripped}'. "
            f"Allowed: {', '.join(config.COMMAND_WHITELIST)}."
        )

    try:
        result = subprocess.run(
            stripped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            return "Command ran with no output."
        if len(output) > 500:
            output = output[:497] + "…"
        return output
    except subprocess.TimeoutExpired:
        return "That command timed out after 30 seconds."
    except Exception as exc:
        return f"Command failed: {exc}"


# ── System info ────────────────────────────────────────────────────────────────

def get_system_info(info_type: str) -> str:
    """
    Return formatted system information.
    info_type: "time" | "date" | "battery" | "cpu" | "ram" | "all"
    """
    logger.info("[TOOL] get_system_info: %s", info_type)

    parts: list[str] = []
    now = datetime.now()

    if info_type in ("time", "all"):
        parts.append(f"Time: {now.strftime('%I:%M %p')}")

    if info_type in ("date", "all"):
        parts.append(f"Date: {now.strftime('%A, %B %d %Y')}")

    if info_type in ("cpu", "all"):
        cpu_pct = psutil.cpu_percent(interval=0.5)
        parts.append(f"CPU: {cpu_pct:.0f}%")

    if info_type in ("ram", "all"):
        ram = psutil.virtual_memory()
        parts.append(
            f"RAM: {ram.used / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB ({ram.percent:.0f}%)"
        )

    if info_type in ("battery", "all"):
        battery = psutil.sensors_battery()
        if battery:
            status = "charging" if battery.power_plugged else "discharging"
            parts.append(f"Battery: {battery.percent:.0f}% ({status})")
        else:
            parts.append("Battery: not available")

    if not parts:
        return f"Unknown info type: '{info_type}'. Use: time, date, battery, cpu, ram, all."

    return " | ".join(parts)


# ── Folder creation ────────────────────────────────────────────────────────────

def create_folder(path: str) -> str:
    """Create a directory (and any intermediate ones) at the given path."""
    logger.info("[TOOL] create_folder: %s", path)
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created at {path}."
    except Exception as exc:
        return f"Failed to create folder at {path}: {exc}"


# ── Keyboard control ───────────────────────────────────────────────────────────

def type_text(text: str) -> str:
    """
    Type text at the current cursor position using the keyboard.
    For text with special characters, uses clipboard-paste method for reliability.
    """
    logger.info("[TOOL] type_text: %r", text[:80])
    try:
        import pyautogui
        import pyperclip

        # Use clipboard paste for speed and Unicode reliability
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        return f"Typed: {text[:80]}{'…' if len(text) > 80 else ''}"
    except ImportError as exc:
        return f"Keyboard control unavailable: {exc}"
    except Exception as exc:
        return f"Failed to type text: {exc}"


def press_hotkey(keys: str) -> str:
    """
    Press a keyboard shortcut.
    keys format: 'ctrl+v', 'alt+tab', 'win+d', 'ctrl+shift+t', 'alt+f4'.
    """
    logger.info("[TOOL] press_hotkey: %s", keys)
    try:
        import pyautogui

        key_list = [k.strip() for k in keys.lower().split("+")]
        pyautogui.hotkey(*key_list)
        return f"Pressed {keys}."
    except ImportError:
        return "pyautogui not installed — keyboard control unavailable."
    except Exception as exc:
        return f"Failed to press {keys}: {exc}"


# ── Volume control ─────────────────────────────────────────────────────────────

def control_volume(action: str, steps: int = 5) -> str:
    """
    Control system audio volume using OS media keys.
    action: 'up' | 'down' | 'mute' | 'unmute'
    steps:  number of key presses (each press ≈ 2% on most Windows systems)
    """
    logger.info("[TOOL] control_volume: action=%s steps=%d", action, steps)
    try:
        import pyautogui

        action = action.lower()
        if action == "up":
            for _ in range(steps):
                pyautogui.press("volumeup")
            return f"Volume raised by {steps} steps, sir."
        elif action == "down":
            for _ in range(steps):
                pyautogui.press("volumedown")
            return f"Volume lowered by {steps} steps, sir."
        elif action in ("mute", "unmute"):
            pyautogui.press("volumemute")
            label = "Muted" if action == "mute" else "Unmuted"
            return f"{label}, sir."
        return f"Unknown volume action: '{action}'. Use: up, down, mute, unmute."
    except ImportError:
        return "pyautogui not installed — volume control unavailable."
    except Exception as exc:
        return f"Volume control failed: {exc}"


# ── Clipboard ──────────────────────────────────────────────────────────────────

def get_clipboard() -> str:
    """Read and return the current clipboard text content."""
    logger.info("[TOOL] get_clipboard")
    try:
        import pyperclip
        content = pyperclip.paste()
        return content if content and content.strip() else "(clipboard is empty)"
    except ImportError:
        return "pyperclip not installed — clipboard access unavailable."
    except Exception as exc:
        return f"Failed to read clipboard: {exc}"


def set_clipboard(text: str) -> str:
    """Copy text to the clipboard."""
    logger.info("[TOOL] set_clipboard: %r", text[:80])
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"Copied to clipboard."
    except ImportError:
        return "pyperclip not installed — clipboard access unavailable."
    except Exception as exc:
        return f"Failed to copy to clipboard: {exc}"
