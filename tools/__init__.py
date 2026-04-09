"""
tools/__init__.py — Tool registry for KOBRA.

Maps tool name strings → callable functions.
Imported by brain.py to dispatch tool calls.

Note: save_memory and recall_memory are NOT here —
they are handled inline in brain.py (they need the Memory instance).
"""

from tools.system import (
    open_app,
    run_command,
    get_system_info,
    create_folder,
    type_text,
    press_hotkey,
    control_volume,
    get_clipboard,
    set_clipboard,
)
from tools.web import open_url, web_search
from tools.media import play_media, stop_media, control_media, play_youtube
from tools.dev import create_file, open_vscode, scaffold_project

TOOL_REGISTRY: dict = {
    # System
    "open_app":         open_app,
    "run_command":      run_command,
    "get_system_info":  get_system_info,
    "create_folder":    create_folder,
    # Keyboard / clipboard
    "type_text":        type_text,
    "press_hotkey":     press_hotkey,
    "control_volume":   control_volume,
    "get_clipboard":    get_clipboard,
    "set_clipboard":    set_clipboard,
    # Media
    "play_media":       play_media,
    "stop_media":       stop_media,
    "control_media":    control_media,
    "play_youtube":     play_youtube,
    # Web
    "open_url":         open_url,
    "web_search":       web_search,
    # Dev
    "create_file":      create_file,
    "open_vscode":      open_vscode,
    "scaffold_project": scaffold_project,
}

__all__ = ["TOOL_REGISTRY"]
