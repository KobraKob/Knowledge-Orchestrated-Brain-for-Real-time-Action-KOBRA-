"""
tools/__init__.py — Tool registry for KOBRA v4.

Maps tool name strings → callable functions.
Imported by brain.py to dispatch tool calls.

Note: save_memory and recall_memory are NOT here —
they are handled inline in brain.py (they need the Memory instance).
"""

from tools.system import (
    open_app,
    install_app,
    run_command,
    get_system_info,
    create_folder,
    type_text,
    press_hotkey,
    control_volume,
    get_clipboard,
    set_clipboard,
    click_at,
    double_click_at,
    right_click_at,
    move_mouse,
    scroll_screen,
    take_screenshot,
)
from tools.web import open_url, web_search
from tools.media import play_media, stop_media, control_media, play_youtube, play_on_spotify
from tools.dev import create_file, open_vscode, scaffold_project

# v4: window management
from tools.window import (
    focus_mode,
    snap_window,
    focus_window,
    close_window,
)

# v4: vision / screen interaction
from tools.screen import read_screen, click_element

TOOL_REGISTRY: dict = {
    # System — launch & install
    "open_app":           open_app,
    "install_app":        install_app,
    "run_command":        run_command,
    "get_system_info":    get_system_info,
    "create_folder":      create_folder,
    # Keyboard / clipboard
    "type_text":          type_text,
    "press_hotkey":       press_hotkey,
    "control_volume":     control_volume,
    "get_clipboard":      get_clipboard,
    "set_clipboard":      set_clipboard,
    # Mouse / screen
    "click_at":           click_at,
    "double_click_at":    double_click_at,
    "right_click_at":     right_click_at,
    "move_mouse":         move_mouse,
    "scroll_screen":      scroll_screen,
    "take_screenshot":    take_screenshot,
    # Media
    "play_media":         play_media,
    "stop_media":         stop_media,
    "control_media":      control_media,
    "play_youtube":       play_youtube,
    "play_on_spotify":    play_on_spotify,
    # Web
    "open_url":           open_url,
    "web_search":         web_search,
    # Dev
    "create_file":        create_file,
    "open_vscode":        open_vscode,
    "scaffold_project":   scaffold_project,
    # Window management (v4)
    "focus_mode":         focus_mode,
    "snap_window":        snap_window,
    "focus_window":       focus_window,
    "close_window":       close_window,
    # Vision / screen interaction (v4)
    "read_screen":        read_screen,
    "click_element":      click_element,
}

__all__ = ["TOOL_REGISTRY"]
