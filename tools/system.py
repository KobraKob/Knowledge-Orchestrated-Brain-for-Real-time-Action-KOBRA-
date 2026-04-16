"""
tools/system.py — System-level tools for KOBRA.

open_app        — launch any application (UWP, Store, .exe, PATH)
install_app     — install software via winget
run_command     — run any shell command (no restrictions — user granted full access)
get_system_info — CPU, RAM, battery, time, date
create_folder   — create a directory
type_text       — type text at the current cursor position
press_hotkey    — send a keyboard shortcut
control_volume  — raise / lower / mute system volume
get_clipboard   — read clipboard content
set_clipboard   — write to clipboard
click_at        — left-click at screen coordinates
double_click_at — double-click at screen coordinates
right_click_at  — right-click at screen coordinates
move_mouse      — move cursor to screen coordinates
scroll_screen   — scroll up or down at current position
take_screenshot — capture the full screen and save to Desktop
"""

import logging
import os
import platform
import subprocess
import time
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)


# ── UWP / Windows Store app protocol map ─────────────────────────────────────
# These apps can't be found as .exe files — they use URI scheme launching.
_UWP_PROTOCOLS: dict[str, str] = {
    "microsoft store":    "ms-windows-store:",
    "store":              "ms-windows-store:",
    "windows store":      "ms-windows-store:",
    "calculator":         "calculator:",
    "settings":           "ms-settings:",
    "windows settings":   "ms-settings:",
    "calendar":           "outlookcal:",
    "mail":               "outlookmail:",
    "maps":               "bingmaps:",
    "bing maps":          "bingmaps:",
    "camera":             "microsoft.windows.camera:",
    "photos":             "ms-photos:",
    "xbox":               "xbox:",
    "onenote":            "onenote:",
    "music":              "mswindowsmusic:",
    "groove":             "mswindowsmusic:",
    "movies":             "mswindowsvideo:",
    "films":              "mswindowsvideo:",
    "news":               "bingnews:",
    "weather":            "bingweather:",
    "people":             "ms-people:",
    "contacts":           "ms-people:",
    "sticky notes":       "ms-stickynotes:",
    "clock":              "ms-clock:",
    "alarms":             "ms-clock:",
    "snipping tool":      "ms-screenclip:",
    "snip":               "ms-screenclip:",
    "screen snip":        "ms-screenclip:",
    "paint":              "mspaint:",
    "xbox game bar":      "ms-gamebar:",
    "game bar":           "ms-gamebar:",
    "feedback hub":       "feedback-hub:",
    "phone link":         "ms-phone:",
    "your phone":         "ms-phone:",
}

# Common EXE aliases for apps not in PATH by their common name
_EXE_ALIASES: dict[str, str] = {
    "chrome":     "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox":    "firefox.exe",
    "edge":       "msedge.exe",
    "microsoft edge": "msedge.exe",
    "brave":      "brave.exe",
    "spotify":    "spotify.exe",
    "discord":    "discord.exe",
    "steam":      "steam.exe",
    "obs":        "obs64.exe",
    "vlc":        "vlc.exe",
    "notepad":    "notepad.exe",
    "notepad++":  "notepad++.exe",
    "word":       "WINWORD.EXE",
    "excel":      "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "outlook":    "OUTLOOK.EXE",
    "teams":      "ms-teams.exe",
    "zoom":       "zoom.exe",
    "task manager": "taskmgr.exe",
    "taskmgr":    "taskmgr.exe",
    "file explorer": "explorer.exe",
    "explorer":   "explorer.exe",
    "cmd":        "cmd.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "terminal":   "wt.exe",
    "windows terminal": "wt.exe",
    "paint 3d":   "mspaint.exe",
    "wordpad":    "wordpad.exe",
    "control panel": "control.exe",
    "device manager": "devmgmt.msc",
    "regedit":    "regedit.exe",
    "msconfig":   "msconfig.exe",
}


# ── App launching ──────────────────────────────────────────────────────────────

def open_app(app_name: str) -> str:
    """
    Launch any application on Windows.
    Strategy:
      0. UWP protocol map  (Microsoft Store, Calculator, Settings …)
      1. EXE alias map     (Chrome, Spotify, Discord …)
      2. Direct Popen      (app is in PATH)
      3. Walk common install dirs  (Programs, AppData …)
      4. os.startfile fallback
    """
    logger.info("[TOOL] open_app: %s", app_name)
    name_lower = app_name.lower().strip()

    # 0. UWP protocol — covers Store apps with no .exe on disk
    for alias, protocol in _UWP_PROTOCOLS.items():
        if alias in name_lower or name_lower in alias:
            try:
                subprocess.Popen(
                    f'start "" "{protocol}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return f"Opened {app_name}, sir."
            except Exception:
                pass

    # 1. EXE alias map
    exe = _EXE_ALIASES.get(name_lower)
    if exe:
        try:
            subprocess.Popen(
                [exe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS,
            )
            return f"Opened {app_name}, sir."
        except (FileNotFoundError, OSError):
            pass

    # 2. Direct launch — works if app is in PATH
    try:
        subprocess.Popen(
            [app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0,
        )
        return f"Opened {app_name}, sir."
    except (FileNotFoundError, OSError):
        pass

    # 3. Windows shell start command (fast — lets Windows resolve the app name)
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", app_name],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Opening {app_name}, sir."
    except Exception:
        pass

    # 3b. PowerShell Start-Process (handles display names and Store apps)
    try:
        result = subprocess.run(
            ["powershell", "-Command", f"Start-Process '{app_name}'"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return f"Opened {app_name}, sir."
    except Exception:
        pass

    # 4. os.startfile — Windows file-association / URI fallback
    try:
        os.startfile(app_name)
        return f"Opened {app_name}, sir."
    except OSError:
        pass

    return (
        f"Could not find '{app_name}' on your system, sir. "
        f"Try install_app to install it first."
    )


# ── App installation ───────────────────────────────────────────────────────────

def install_app(app_name: str) -> str:
    """
    Install a Windows application using winget (Windows Package Manager).
    Works for most apps: Spotify, Chrome, VLC, Discord, 7-Zip, etc.
    Runs silently and accepts all agreements automatically.
    """
    logger.info("[TOOL] install_app: %s", app_name)
    try:
        # Run winget install — silent mode, auto-accept agreements
        cmd = (
            f'winget install "{app_name}" '
            f'--accept-package-agreements --accept-source-agreements --silent'
        )
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=180,   # Some installs take a while
        )
        output = (result.stdout + result.stderr).strip()

        if result.returncode == 0:
            return f"Successfully installed {app_name}, sir. You can now open it."

        # winget exit code -1978335212 means "already installed"
        if result.returncode == -1978335212 or "already installed" in output.lower():
            return f"{app_name} is already installed on your system, sir."

        return (
            f"Install attempt for '{app_name}' completed. "
            f"Output: {output[:200] if output else 'No output.'}"
        )

    except subprocess.TimeoutExpired:
        return (
            f"Installation of '{app_name}' is still running in the background, sir. "
            f"It should complete shortly."
        )
    except Exception as exc:
        return f"Failed to install '{app_name}': {exc}"


# ── Shell commands ─────────────────────────────────────────────────────────────

def run_command(command: str) -> str:
    """
    Run any shell command with full system access.
    Commands that open GUI windows (start, explorer, etc.) are launched detached.
    Output is capped at 500 chars to keep TTS responses concise.
    """
    logger.info("[TOOL] run_command: %s", command)
    stripped = command.strip()
    if not stripped:
        return "No command provided."

    # Detect GUI / window-launching commands that should not block
    first_token = stripped.split()[0].lower()
    gui_launchers = {"start", "explorer", "msiexec", "winget", "ms-windows-store:"}
    is_gui = first_token in gui_launchers or stripped.lower().startswith("start ")

    try:
        if is_gui:
            # Fire-and-forget — don't wait for GUI app to close
            subprocess.Popen(
                stripped,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return f"Launched: {stripped[:80]}, sir."

        result = subprocess.run(
            stripped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            return "Command ran with no output, sir."
        if len(output) > 500:
            output = output[:497] + "…"
        return output
    except subprocess.TimeoutExpired:
        return "That command timed out after 60 seconds, sir."
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
        return f"Folder created at {path}, sir."
    except Exception as exc:
        return f"Failed to create folder at {path}: {exc}"


# ── Keyboard control ───────────────────────────────────────────────────────────

def type_text(text: str) -> str:
    """
    Type text at the current cursor position.
    Uses clipboard-paste for Unicode reliability.
    """
    logger.info("[TOOL] type_text: %r", text[:80])
    try:
        import pyautogui
        import pyperclip

        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        return f"Typed text, sir."
    except ImportError as exc:
        return f"Keyboard control unavailable: {exc}"
    except Exception as exc:
        return f"Failed to type text: {exc}"


def press_hotkey(keys: str) -> str:
    """
    Press a keyboard shortcut.
    Format: 'ctrl+v', 'alt+tab', 'win+d', 'ctrl+shift+t', 'alt+f4', 'win', 'enter'.
    """
    logger.info("[TOOL] press_hotkey: %s", keys)
    try:
        import pyautogui

        key_list = [k.strip() for k in keys.lower().split("+")]
        if len(key_list) == 1:
            pyautogui.press(key_list[0])
        else:
            pyautogui.hotkey(*key_list)
        return f"Pressed {keys}, sir."
    except ImportError:
        return "pyautogui not installed — keyboard control unavailable."
    except Exception as exc:
        return f"Failed to press {keys}: {exc}"


# ── Volume control ─────────────────────────────────────────────────────────────

def control_volume(action: str, steps: int = 5) -> str:
    """
    Control system audio volume using OS media keys.
    action: 'up' | 'down' | 'mute' | 'unmute'
    steps:  number of key presses (each ≈ 2% volume on Windows)
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
        return "Copied to clipboard, sir."
    except ImportError:
        return "pyperclip not installed — clipboard access unavailable."
    except Exception as exc:
        return f"Failed to copy to clipboard: {exc}"


# ── Mouse control ──────────────────────────────────────────────────────────────

def click_at(x: int, y: int) -> str:
    """
    Left-click at specific screen coordinates (pixels from top-left corner).
    Use take_screenshot first to see the screen and determine coordinates.
    """
    logger.info("[TOOL] click_at: (%d, %d)", x, y)
    try:
        import pyautogui
        pyautogui.click(x, y)
        return f"Clicked at ({x}, {y}), sir."
    except ImportError:
        return "pyautogui not installed — mouse control unavailable."
    except Exception as exc:
        return f"Click failed: {exc}"


def double_click_at(x: int, y: int) -> str:
    """Double-click at specific screen coordinates."""
    logger.info("[TOOL] double_click_at: (%d, %d)", x, y)
    try:
        import pyautogui
        pyautogui.doubleClick(x, y)
        return f"Double-clicked at ({x}, {y}), sir."
    except ImportError:
        return "pyautogui not installed — mouse control unavailable."
    except Exception as exc:
        return f"Double-click failed: {exc}"


def right_click_at(x: int, y: int) -> str:
    """Right-click at specific screen coordinates."""
    logger.info("[TOOL] right_click_at: (%d, %d)", x, y)
    try:
        import pyautogui
        pyautogui.rightClick(x, y)
        return f"Right-clicked at ({x}, {y}), sir."
    except ImportError:
        return "pyautogui not installed — mouse control unavailable."
    except Exception as exc:
        return f"Right-click failed: {exc}"


def move_mouse(x: int, y: int) -> str:
    """Move the mouse cursor to specific screen coordinates without clicking."""
    logger.info("[TOOL] move_mouse: (%d, %d)", x, y)
    try:
        import pyautogui
        pyautogui.moveTo(x, y, duration=0.2)
        return f"Mouse moved to ({x}, {y}), sir."
    except ImportError:
        return "pyautogui not installed — mouse control unavailable."
    except Exception as exc:
        return f"Mouse move failed: {exc}"


def scroll_screen(direction: str, amount: int = 5) -> str:
    """
    Scroll the screen at the current mouse position.
    direction: 'up' | 'down'
    amount: number of scroll steps (default 5)
    """
    logger.info("[TOOL] scroll_screen: direction=%s amount=%d", direction, amount)
    try:
        import pyautogui
        clicks = amount if direction.lower() == "up" else -amount
        pyautogui.scroll(clicks)
        return f"Scrolled {direction} by {amount} steps, sir."
    except ImportError:
        return "pyautogui not installed — scroll unavailable."
    except Exception as exc:
        return f"Scroll failed: {exc}"


def take_screenshot() -> str:
    """
    Capture the full screen and save it to the Desktop.
    Returns the file path — use this to see what's on screen before clicking.
    """
    logger.info("[TOOL] take_screenshot")
    try:
        import pyautogui
        username = os.environ.get("USERNAME", "user")
        desktop = f"C:\\Users\\{username}\\Desktop"
        os.makedirs(desktop, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(desktop, f"kobra_screenshot_{timestamp}.png")
        pyautogui.screenshot(path)
        return f"Screenshot saved to {path}, sir."
    except ImportError:
        return "pyautogui not installed — screenshot unavailable."
    except Exception as exc:
        return f"Screenshot failed: {exc}"
